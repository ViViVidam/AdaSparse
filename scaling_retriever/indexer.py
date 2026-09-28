import os
import sys
from typing import List, Tuple
import pickle
import logging
from collections import defaultdict
import json
import gzip
import torch 
from tqdm import tqdm
from transformers.modeling_utils import unwrap_model
import numpy as np
import ujson
import faiss 
import numba
from numba.typed import Dict
from numba import types
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

from scaling_retriever.utils.utils import is_first_worker, to_list, obtain_doc_vec_dir_files, supports_bfloat16
from scaling_retriever.utils.inverted_index import IndexDictOfArray
from scaling_retriever.modeling.losses.regulariaztion import L0

logger = logging.getLogger()

def alpha_prune(reps, alpha):
    """Keep the fewest top values per row whose cumulative sum reaches `alpha` fraction of the L1 sum."""
    if alpha >= 1:
        # keep 100% of the mass == keep everything; also avoids fp issues where
        # cumsum never reaches l1_sum and argmax over all-False collapses to 0
        return reps
    # compute in float32: under bf16 autocast the rounding of sum vs cumsum
    # differs enough to move the cutoff
    sorted_vals, sorted_idx = torch.sort(reps.float(), dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_vals, dim=-1)
    l1_sum = cumsum[..., -1:]
    reached = cumsum >= alpha * l1_sum
    # fallback: if fp error keeps the threshold from ever being reached,
    # keep all non-zero entries instead of argmax's default 0 (top-1 only)
    cutoff = torch.where(reached.any(dim=-1),
                         reached.long().argmax(dim=-1),
                         (sorted_vals > 0).sum(dim=-1) - 1)
    positions = torch.arange(reps.size(-1), device=reps.device)
    keep_mask = positions.unsqueeze(0) <= cutoff.unsqueeze(1)
    pruned = torch.zeros_like(reps)
    pruned.scatter_(-1, sorted_idx, (sorted_vals * keep_mask).to(reps.dtype))
    return pruned

def store_embs(model, collection_loader, local_rank, index_dir, device,
               chunk_size=2_000_000, use_fp16=False, is_query=False, idx_to_id=None):
    write_freq = chunk_size // collection_loader.batch_size
    if is_first_worker():
        print("write_freq: {}, batch_size: {}, chunk_size: {}".format(write_freq, collection_loader.batch_size, chunk_size))
    if supports_bfloat16():
        dtype = torch.bfloat16
        print("Using bfloat16")
    else:
        dtype = torch.float32
        print("Using float32")
    
    
    embeddings = []
    embeddings_ids = []
    
    chunk_idx = 0
    for idx, batch in tqdm(enumerate(collection_loader), disable= not is_first_worker(), 
                                desc=f"encode # {len(collection_loader)} seqs",
                            total=len(collection_loader)):
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=dtype):
                inputs = {k:v.to(device) for k, v in batch.items() if k != "ids"}
                if is_query:
                    raise NotImplementedError
                else:
                    reps = unwrap_model(model).doc_encode(**inputs)
                #reps = model(**inputs)            
                text_ids = batch["ids"] #.tolist()

        embeddings.append(reps.float().cpu().numpy())
        assert isinstance(text_ids, list)
        embeddings_ids.extend(text_ids)

        if (idx + 1) % write_freq == 0:
            embeddings = np.concatenate(embeddings)
            if isinstance(embeddings_ids[0], int):
                embeddings_ids = np.array(embeddings_ids, dtype=np.int64)
            assert len(embeddings) == len(embeddings_ids), (len(embeddings), len(embeddings_ids))

            text_path = os.path.join(index_dir, "embs_{}_{}.npy".format(local_rank, chunk_idx))
            id_path = os.path.join(index_dir, "ids_{}_{}.npy".format(local_rank, chunk_idx))
            np.save(text_path, embeddings)
            np.save(id_path, embeddings_ids)

            del embeddings, embeddings_ids
            embeddings, embeddings_ids = [], []

            chunk_idx += 1 
    
    if len(embeddings) != 0:
        embeddings = np.concatenate(embeddings)
        if isinstance(embeddings_ids[0], int):
            embeddings_ids = np.array(embeddings_ids, dtype=np.int64)
        assert len(embeddings) == len(embeddings_ids), (len(embeddings), len(embeddings_ids))
        print("last embedddings shape = {}".format(embeddings.shape))
        text_path = os.path.join(index_dir, "embs_{}_{}.npy".format(local_rank, chunk_idx))
        id_path = os.path.join(index_dir, "ids_{}_{}.npy".format(local_rank, chunk_idx))
        np.save(text_path, embeddings)
        np.save(id_path, embeddings_ids)

        del embeddings, embeddings_ids
        chunk_idx += 1 
        
    plan = {"nranks": torch.distributed.get_world_size(), 
            "num_chunks": chunk_idx, 
            "index_path": os.path.join(index_dir, "model.index")}
    print("plan: ", plan)
    
    if is_first_worker():
        with open(os.path.join(index_dir, "plan.json"), "w") as fout:
            ujson.dump(plan, fout)


def process_query_for_multiprocess(args):
    """
    Function to process a single query in parallel.
    """
    qid, col, values, threshold, topk, \
        doc_ids, size_collection, numba_score_float, select_topk = args
    res = defaultdict(dict)
    stats = defaultdict(float)

    # Perform sparse retrieval for the query
    filtered_indexes, scores = numba_score_float(
        shared_doc_ids,
        shared_doc_values,
        col,
        values,
        threshold=threshold,
        size_collection=size_collection,
    )
    filtered_indexes, scores = select_topk(filtered_indexes, scores, k=topk)

    for id_, sc in zip(filtered_indexes, scores):
        res[str(qid)][str(doc_ids[id_])] = float(sc)
    stats["L0_q"] = len(values)

    return res, stats

        
class DenseIndexer(object):
    def __init__(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        self.index_id_to_db_id = []
        self.index = None

    def init_index(self, vector_sz: int):
        raise NotImplementedError

    def index_data(self, data: List[Tuple[object, np.array]]):
        raise NotImplementedError

    def get_index_name(self):
        raise NotImplementedError

    def search_knn(self, query_vectors: np.array, top_docs: int):
        raise NotImplementedError
    
    def serialize(self, file: str):
        logger.info("Serializing index to %s", file)

        if os.path.isdir(file):
            os.makedirs(file, exist_ok=True)
            index_file = os.path.join(file, "index.dpr")
            meta_file = os.path.join(file, "index_meta.dpr")
        else:
            index_file = file + ".index.dpr"
            meta_file = file + ".index_meta.dpr"

        faiss.write_index(self.index, index_file)
        with open(meta_file, mode="wb") as f:
            pickle.dump(self.index_id_to_db_id, f)

    def get_files(self, path: str):
        if os.path.isdir(path):
            index_file = os.path.join(path, "index.dpr")
            meta_file = os.path.join(path, "index_meta.dpr")
        else:
            index_file = path + ".{}.dpr".format(self.get_index_name())
            meta_file = path + ".{}_meta.dpr".format(self.get_index_name())
        return index_file, meta_file

    def index_exists(self, path: str):
        index_file, meta_file = self.get_files(path)
        return os.path.isfile(index_file) and os.path.isfile(meta_file)

    def deserialize(self, path: str):
        logger.info("Loading index from %s", path)
        index_file, meta_file = self.get_files(path)

        self.index = faiss.read_index(index_file)
        logger.info("Loaded index of type %s and size %d", type(self.index), self.index.ntotal)

        with open(meta_file, "rb") as reader:
            self.index_id_to_db_id = pickle.load(reader)
        assert (
            len(self.index_id_to_db_id) == self.index.ntotal
        ), "Deserialized index_id_to_db_id should match faiss index size"

    def _update_id_mapping(self, db_ids: List):
        self.index_id_to_db_id.extend(db_ids)
        return len(self.index_id_to_db_id)
    
    
class DenseFlatIndexer(DenseIndexer):
    def __init__(self, buffer_size: int = 50000):
        super().__init__(buffer_size)
        
    def init_index(self, hidden_dim):
        self.index = faiss.IndexFlatIP(hidden_dim)
        
    def index_data(self, doc_reps, doc_ids):
        assert len(doc_reps) == len(doc_ids)
        n = len(doc_reps)
        
        for i in tqdm(range(0, n, self.buffer_size), total=n // self.buffer_size, desc="indexing"):
            self.index.add(doc_reps[i:i+self.buffer_size])
            n_total = self._update_id_mapping(doc_ids[i:i+self.buffer_size])
            logger.info("data indexed %d", n_total)
        
        assert n_total == n, (n_total, n)
        logger.info("total data indexed %d", n_total)
        
    def search_knn(self, query_reps: np.array, top_docs: int):
        scores, indexes = self.index.search(query_reps, k=top_docs)
        top_doc_ids = [[self.index_id_to_db_id[idx] for idx in per_query_indexes] for per_query_indexes in indexes]

        return top_doc_ids, scores
    
    def get_index_name(self):
        return "flat_index"
    
        
class SparseIndexer:
    def __init__(self, model, index_dir, tokenizer, device, compute_stats=False, dim_voc=None, force_new=True,
                 filename="array_index.h5py", **kwargs):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.index_dir = index_dir
        self.sparse_index = IndexDictOfArray(self.index_dir, dim_voc=dim_voc, force_new=force_new, filename=filename)
        self.compute_stats = compute_stats
        self.device = device
        if self.compute_stats:
            self.l0 = L0()
        
        self.model.to(self.device)
        self.local_rank = self.device
        assert self.local_rank == torch.distributed.get_rank(), (self.local_rank, torch.distributed.get_rank())
        self.world_size = torch.distributed.get_world_size()
        print("world_size: {}, local_rank: {}".format(self.world_size, self.local_rank))
    
    def index(self, collection_loader, id_dict=None, quant=False, output_json=False, alpha=None, ciff_term_ids=False):
        if supports_bfloat16():
            dtype = torch.bfloat16
            print("Using bfloat16")
        else:
            dtype = torch.float32
            print("Using float32")

        if output_json:
            fo = gzip.open(os.path.join(self.index_dir,"index.jsonl.gz"),"wb")
        # we change doc_ids from list to dict to support muliti-gpu indexing
        doc_ids = {}
        total_nonzeros = 0  # non-zero entries actually written to the index on this rank
        quant_nonzeros = 0  # non-zero entries the index would hold after quantization
        if self.compute_stats:
            stats = defaultdict(float)
        count = 0
        first_batch_ids = None
        with torch.inference_mode():
            for t, batch in enumerate(tqdm(collection_loader, disable = not is_first_worker())):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in {"ids","texts"}}
                with torch.amp.autocast("cuda", dtype=dtype):
                    batch_documents = self.model.doc_encode(**inputs) #[bz, vocab_size]
                if alpha is not None:
                    batch_documents = alpha_prune(batch_documents, alpha)
                if quant:
                    int_batch_documents = torch.round(batch_documents * 100).to(torch.int32)
                    quanted_batch_documents = torch.clamp(int_batch_documents,min=0,max=255)
                    # over_float += (quanted_batch_documents.to(torch.int32) != quanted).sum()
                    quant_nonzeros += torch.count_nonzero(quanted_batch_documents).item()
                    if self.compute_stats:
                        stats["quanted_L0_d"] += self.l0(quanted_batch_documents).item()
                        stats["overflow"] += (quanted_batch_documents != int_batch_documents).sum().item()
                else:
                    quanted_batch_documents = batch_documents
                if self.compute_stats:
                    stats["L0_d"] += self.l0(batch_documents).item()
                if isinstance(batch["ids"], torch.Tensor):
                    batch_ids = to_list(batch["ids"])
                else:
                    assert isinstance(batch["ids"], list)
                    batch_ids = batch["ids"]
                if output_json:
                    if t == 0:
                        # DistributedSampler pads the tail of each rank's shard by repeating
                        # indices[0:padding_size] (padding_size <= world_size-1), so duplicates
                        # always come from the very start of iteration and appear at the very end.
                        # Remembering only the first batch covers all possible padding candidates
                        # in O(batch_size) memory rather than O(corpus_size).
                        first_batch_ids = set(batch_ids)
                    # ciff output mirrors what quantization would put in the index:
                    # quantized weights when quant is on, raw float weights otherwise
                    for idx, item in enumerate(zip(batch_ids, quanted_batch_documents)):
                        did, doc_rep = item
                        if t > 0 and did in first_batch_ids:
                            continue
                        col = torch.nonzero(doc_rep).squeeze(-1).cpu().tolist()
                        if quant:
                            weights = doc_rep[col].cpu().tolist()
                        else:
                            weights = doc_rep[col].float().cpu().tolist()
                        if ciff_term_ids:
                            keys = [str(k) for k in col]
                        else:
                            keys = self.tokenizer.convert_ids_to_tokens(col)
                        d = {k: v for k, v in zip(keys, weights)}
                        outline = json.dumps({"id": did, "content": batch["texts"][idx], "vector": d}) + "\n"
                        fo.write(outline.encode('utf-8'))
                row, col = torch.nonzero(batch_documents, as_tuple=True) # This has some problem, the values are float
                data = batch_documents[row, col]
                row = row + count
                total_nonzeros += len(row)
                g_row = row.cpu().numpy() * self.world_size + self.local_rank
                if id_dict:
                    batch_ids = [id_dict[x] for x in batch_ids]
                #print(len(row)/len(batch_ids))
                unique_g_row = np.sort(np.unique(g_row))
                if len(unique_g_row) == len(batch_ids):
                    doc_ids.update({int(x): y for x, y in zip(unique_g_row, batch_ids)})
                else:
                    # there exists slim chance that some of docid don't have any posting list
                    assert len(unique_g_row) < len(batch_ids), (len(unique_g_row), len(batch_ids))
                    all_idxes = count + np.arange(len(batch_ids))
                    all_idxes = all_idxes * self.world_size + self.local_rank
                    for _i, _idx in enumerate(all_idxes):
                        if _idx in unique_g_row:
                            doc_ids[_idx] = batch_ids[_i]
                    
                    print(all_idxes, unique_g_row)                    
                self.sparse_index.add_batch_document(g_row, col.cpu().numpy(), data.float().cpu().numpy(), n_docs=len(batch_ids)) 
                count += len(batch_ids)

        print(f"[rank {self.local_rank}] non-zero entries in index: {total_nonzeros} "
              f"over {count} docs (avg {total_nonzeros/count:.2f} per doc)", flush=True)
        if quant:
            print(f"[rank {self.local_rank}] non-zero entries after quantization: {quant_nonzeros} "
                  f"over {count} docs (avg {quant_nonzeros/count:.2f} per doc)", flush=True)
        # print(f"over flow numbers: {over_float/count}")

        if self.compute_stats:
            stats = {key: value / len(collection_loader) for key, value in stats.items()}
            # exact totals (not batch-averaged) so they can be cross-checked
            # against the printed numbers and against the merged index
            stats["total_nonzero_entries"] = total_nonzeros
            stats["n_docs"] = count
            if quant:
                stats["quanted_total_nonzero_entries"] = quant_nonzeros
            with open(os.path.join(self.index_dir, "index_stats.json"), "w") as handler:
                ujson.dump(stats, handler)
        if output_json:
            fo.close()
        if self.index_dir is not None:
            self.sparse_index.save()
            pickle.dump(doc_ids, open(os.path.join(self.index_dir, "doc_ids.pkl"), "wb"))
            print("done iterating over the corpus...")
            print("index contains {} posting lists".format(len(self.sparse_index)))
            print("index contains {} documents".format(len(doc_ids)))
        else:
            # if no index_dir, we do not write the index to disk but return it
            for key in list(self.sparse_index.index_doc_id.keys()):
                # convert to numpy
                self.sparse_index.index_doc_id[key] = np.array(self.sparse_index.index_doc_id[key], dtype=np.int32)
                self.sparse_index.index_doc_value[key] = np.array(self.sparse_index.index_doc_value[key],
                                                                  dtype=np.float32)
            out = {"index": self.sparse_index, "ids_mapping": doc_ids}
            if self.compute_stats:
                out["stats"] = stats
            return out
  
        
class SparseQueryEncoder:
    def __init__(self, model, device,tokenizer=None):
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.device = device
        self.model.to(device)
        self.l0 = L0()

    def generate_query_vecs(self, q_loader,quant=False, alpha=None, ciff_term_ids=False):
        sparse_query_vecs = []
        ciff_lines = []
        qids = []
        nonzero_cnt, vectors_cnt = 0, 0
        overflow_cnt = 0
        with torch.inference_mode():
            for t, batch in enumerate(tqdm(q_loader, total=len(q_loader),
                                      desc="generate query vecs", disable = not is_first_worker())):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in {"ids","texts"}}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16 if supports_bfloat16() else torch.float32):
                    batch_sparse_reps = self.model.query_encode(**inputs)
                if alpha is not None:
                    batch_sparse_reps = alpha_prune(batch_sparse_reps, alpha)
                qids.extend(batch["ids"] if isinstance(batch["ids"], list) else to_list(batch["ids"]))
                for idx,sparse_rep in enumerate(batch_sparse_reps):
                    sparse_rep = sparse_rep.unsqueeze(0)
                    # ciff output: quantized weights when quant is on, raw floats otherwise
                    if quant:
                        quanted = torch.round(sparse_rep*100).to(torch.int32)
                        sparse_rep_ciff = torch.clamp(quanted,0,255)
                        overflow_cnt += (quanted != sparse_rep_ciff).sum()
                        ciff_np = sparse_rep_ciff.squeeze(0).cpu().numpy()
                    else:
                        ciff_np = sparse_rep.squeeze(0).float().cpu().numpy()
                    ciff_col = np.nonzero(ciff_np)[0]
                    weights = ciff_np[ciff_col].tolist()
                    if ciff_term_ids:
                        keys = [str(int(k)) for k in ciff_col]
                    else:
                        keys = self.tokenizer.convert_ids_to_tokens([int(k) for k in ciff_col])
                    d = {k: v for k, v in zip(keys, weights)}
                    outline = json.dumps({"id": batch['ids'][idx], "content": "", "vector": d}) + "\n"
                    ciff_lines.append(outline)
                    row, col = torch.nonzero(sparse_rep, as_tuple=True)
                    assert all(row == 0), row
                    data = sparse_rep[row, col]
                    sparse_query_vecs.append((col.cpu().numpy().astype(np.int32),
                                              data.cpu().numpy().astype(np.float32)))
                    nonzero_cnt += len(col)
                    vectors_cnt += 1
        L0_d = nonzero_cnt / vectors_cnt
        return sparse_query_vecs, ciff_lines, qids, L0_d, overflow_cnt / vectors_cnt

    def generate_query_vecs_save(self, q_loader, out_dir,quant=False,out_ciff=False, alpha=None, ciff_term_ids=False):
        if not out_ciff:
            print("query encoding is only for producing ciff output (queries.jsonl); "
                  "for evaluation please run eval_sparse.sh instead", flush=True)
            sys.exit(1)
        vecs, ciff_lines, qids, L0_d, overflow_cnt = self.generate_query_vecs(
            q_loader, quant=quant, alpha=alpha, ciff_term_ids=ciff_term_ids)
        with open(os.path.join(out_dir, "queries.jsonl"),"w+") as fout:
            for dump_str in ciff_lines:
                fout.write(dump_str)
        with open(os.path.join(out_dir, "stats.txt"), "w") as f_l0d:
            f_l0d.write("average length: "+str(L0_d)+"\n")
            f_l0d.write(f"overflow: {overflow_cnt}")

        
class SparseRetrieval:
    """retrieval from SparseIndexing
    """
    @staticmethod
    def select_topk(filtered_indexes, scores, k):
        if len(filtered_indexes) > k:
            sorted_ = np.argpartition(scores, k)[:k]
            filtered_indexes, scores = filtered_indexes[sorted_], -scores[sorted_]
        else:
            scores = -scores
        return filtered_indexes, scores

    @staticmethod
    @numba.njit(nogil=True, parallel=True, cache=True)
    def numba_score_float(inverted_index_ids: numba.typed.Dict,
                          inverted_index_floats: numba.typed.Dict,
                          indexes_to_retrieve: np.ndarray,
                          query_values: np.ndarray,
                          threshold: float,
                          size_collection: int):
        scores = np.zeros(size_collection, dtype=np.float32)  # initialize array with size = size of collection
        n = len(indexes_to_retrieve)
        for _idx in range(n):
            local_idx = indexes_to_retrieve[_idx]  # which posting list to search
            query_float = query_values[_idx]  # what is the value of the query for this posting list
            retrieved_indexes = inverted_index_ids[local_idx]  # get indexes from posting list
            retrieved_floats = inverted_index_floats[local_idx]  # get values from posting list
            for j in numba.prange(len(retrieved_indexes)):
                #print(retrieved_indexes[j], retrieved_floats[j])
                scores[retrieved_indexes[j]] += query_float * retrieved_floats[j]
        filtered_indexes = np.argwhere(scores > threshold)[:, 0]  # ideally we should have a threshold to filter
        # unused documents => this should be tuned, currently it is set to 0
        return filtered_indexes, -scores[filtered_indexes]

    @staticmethod
    @numba.njit(nogil=True, parallel=True, cache=True)
    def numba_score_int8(inverted_index_ids: numba.typed.Dict,
                         inverted_index_floats: numba.typed.Dict,
                         indexes_to_retrieve: np.ndarray,
                         query_values: np.ndarray,
                         threshold: float,
                         size_collection: int):
        scores = np.zeros(size_collection, dtype=np.int32)
        n = len(indexes_to_retrieve)
        for _idx in range(n):
            local_idx = indexes_to_retrieve[_idx]
            q_int = np.int32(query_values[_idx])  # already quantized in _generate_query_vecs
            retrieved_indexes = inverted_index_ids[local_idx]
            retrieved_floats = inverted_index_floats[local_idx]
            for j in numba.prange(len(retrieved_indexes)):
                d_int = np.int32(min(max(round(np.float64(retrieved_floats[j]) * 100.0), 0), 255))
                scores[retrieved_indexes[j]] += q_int * d_int
        filtered_indexes = np.argwhere(scores > np.int32(0))[:, 0]
        return filtered_indexes, -scores[filtered_indexes]

    def __init__(self, model, config, dim_voc, device, tokenizer= None, dataset_name=None, index_d=None, compute_stats=False, is_beir=False,
                 **kwargs):
        self.model = model 
        self.model.eval()
        self.tokenizer = tokenizer
        assert ("index_dir" in config and index_d is None) or (
                "index_dir" not in config and index_d is not None)
        if "index_dir" in config:
            self.sparse_index = IndexDictOfArray(config["index_dir"], dim_voc=dim_voc)
            self.doc_ids = pickle.load(open(os.path.join(config["index_dir"], "doc_ids.pkl"), "rb"))
        else:
            self.sparse_index = index_d["index"]
            self.doc_ids = index_d["ids_mapping"]
            for i in range(dim_voc):
                # missing keys (== posting lists), causing issues for retrieval => fill with empty
                if i not in self.sparse_index.index_doc_id:
                    self.sparse_index.index_doc_id[i] = np.array([], dtype=np.int32)
                    self.sparse_index.index_doc_value[i] = np.array([], dtype=np.float32)
        # convert to numba
        self.numba_index_doc_ids = numba.typed.Dict()
        self.numba_index_doc_values = numba.typed.Dict()
        for key, value in self.sparse_index.index_doc_id.items():
            self.numba_index_doc_ids[key] = value
        for key, value in self.sparse_index.index_doc_value.items():
            self.numba_index_doc_values[key] = value
        
        self.out_dir = os.path.join(config["out_dir"], dataset_name) if (dataset_name is not None and not is_beir) \
            else config["out_dir"]
        self.doc_stats = index_d["stats"] if (index_d is not None and compute_stats) else None
        self.compute_stats = compute_stats
        if self.compute_stats:
            self.l0 = L0()

        self.device = device
        self.model.to(device)
        
    def _generate_query_vecs(self, q_loader,quant=False, alpha=None):
        sparse_query_vecs = []
        qids = []
        q_avg_len = 0
        with torch.inference_mode():
            for t, batch in enumerate(tqdm(q_loader, total=len(q_loader),
                                      desc="generate query vecs", disable = not is_first_worker())):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in {"ids","texts"}}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16 if supports_bfloat16() else torch.float32):
                    batch_sparse_reps = self.model.query_encode(**inputs)
                if alpha is not None:
                    batch_sparse_reps = alpha_prune(batch_sparse_reps, alpha)
                q_avg_len += torch.count_nonzero(batch_sparse_reps)
                qids.extend(batch["ids"] if isinstance(batch["ids"], list) else to_list(batch["ids"]))
                for sparse_rep in batch_sparse_reps:
                    sparse_rep = sparse_rep.unsqueeze(0)
                    row, col = torch.nonzero(sparse_rep, as_tuple=True)
                    assert all(row == 0), row
                    data = sparse_rep[row, col]

                    data_np = data.cpu().numpy()
                    if quant:
                        data_np = np.clip(np.round(data_np * 100), 0, 255).astype(np.uint8)
                    else:
                        data_np = data_np.astype(np.float32)
                    sparse_query_vecs.append((col.cpu().numpy().astype(np.int32), data_np))
        print(f"generated {len(qids)} query vecs, avg query length (non-zero entries): "
              f"{q_avg_len.item()/len(qids):.2f}", flush=True)
        return sparse_query_vecs, qids

    def _sparse_retrieve_multithreaded(self, sparse_query_vecs, qids, threshold=0., topk=1000, quant=False):
        """
        Parallelized sparse retrieval function.
        """
        score_fn = self.numba_score_int8 if quant else self.numba_score_float

        def _process_query_threaded(qid, col, values, threshold,
                                topk, inverted_index_ids, inverted_index_floats,
                                doc_ids, size_collection, score_fn, select_topk):
            """
            Function to process a single query in a threaded environment.
            """
            res = defaultdict(dict)
            stats = defaultdict(float)

            # Perform sparse retrieval for the query
            filtered_indexes, scores = score_fn(
                inverted_index_ids,
                inverted_index_floats,
                col,
                values,
                threshold=threshold,
                size_collection=size_collection,
            )
            filtered_indexes, scores = select_topk(filtered_indexes, scores, k=topk)

            for id_, sc in zip(filtered_indexes, scores):
                res[str(qid)][str(doc_ids[id_])] = float(sc)
            stats["L0_q"] = len(values)

            return res, stats

        # Prepare arguments for multiprocessing
        size_collection = self.sparse_index.nb_docs()
        args_list = [
            (
                qid,
                col,
                values,
                threshold,
                topk,
                self.numba_index_doc_ids,
                self.numba_index_doc_values,
                self.doc_ids,
                size_collection,
                score_fn,
                self.select_topk
            )
            for qid, (col, values) in zip(qids, sparse_query_vecs)
        ]

         # Use ThreadPoolExecutor for multithreading
        res = defaultdict(dict)
        stats = defaultdict(float)
        
        # set to 16 will make my retrieval extremly slow in some cases.
        with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust max_workers as needed
            futures = [
                executor.submit(_process_query_threaded, *args)
                for args in args_list
            ]

            # Track progress with tqdm
            for future in tqdm(as_completed(futures), total=len(futures), desc="retrieval by inverted index", disable=not is_first_worker()):
                r, s = future.result()
                for qid, docs in r.items():
                    res[qid].update(docs)
                for k, v in s.items():
                    stats[k] += v / len(qids)


        return res, stats
    
    def _sparse_retrieve_multiprocess(self, sparse_query_vecs, qids, threshold=0., topk=1000):
        """
        Parallelized sparse retrieval function.
        """
        # Global variable for shared data 
        shared_doc_ids = None 
        shared_doc_values = None

        def init_worker(doc_ids, doc_values):
            global shared_doc_ids, shared_doc_values
            shared_doc_ids = doc_ids
            shared_doc_values = doc_values
        
        # Prepare arguments for multiprocessing
        size_collection = self.sparse_index.nb_docs()
        args_list = [
            (
                qid,
                col,
                values,
                threshold,
                topk,
                self.doc_ids,
                size_collection,
                self.numba_score_float,
                self.select_topk
            )
            for qid, (col, values) in zip(qids, sparse_query_vecs)
        ]

        # Use multiprocessing to parallelize
        with mp.Pool(processes=mp.cpu_count(),
                     initializer=init_worker,
                     initargs=(self.numba_index_doc_ids, self.numba_index_doc_values)) as pool:
            results = list(
                tqdm(
                    pool.imap(process_query_for_multiprocess, args_list),
                    desc="retrieval by inverted index",
                    total=len(qids),
                    disable=not is_first_worker()
                )
            )

        # Aggregate results
        res = defaultdict(dict)
        stats = defaultdict(float)
        for r, s in results:
            for qid, docs in r.items():
                res[qid].update(docs)
            for k, v in s.items():
                stats[k] += v / len(qids)

        return res, stats
    
    def retrieve(self, q_loader, topk, quant=False, threshold=0., alpha=None):
        sparse_query_vecs, qids = self._generate_query_vecs(q_loader, quant=quant, alpha=alpha)
        res, stats = self._sparse_retrieve_multithreaded(sparse_query_vecs, qids, threshold=threshold, topk=topk, quant=quant)
        
        if self.compute_stats:
            with open(os.path.join(self.out_dir, "q_stats.json"), "w") as handler:
                json.dump(stats, handler)
        with open(os.path.join(self.out_dir, "run.json"), "w") as handler:
            json.dump(res, handler)
        
        return res

    def old_retrieve(self, q_loader, top_k, name=None, return_d=False, id_dict=False, threshold=0):
        if supports_bfloat16():
            dtype = torch.bfloat16
            print("Using bfloat16")
        else:
            dtype = torch.float32
            print("Using float32")
            
        os.makedirs(self.out_dir, exist_ok=True)
        if self.compute_stats:
            os.makedirs(os.path.join(self.out_dir, "stats"), exist_ok=True)
        res = defaultdict(dict)
        if self.compute_stats:
            stats = defaultdict(float)
        with torch.inference_mode():
            for t, batch in enumerate(tqdm(q_loader)):
                #q_id = to_list(batch["ids"])[0]
                q_id = batch["ids"][0]
                assert len(batch["ids"]) == 1, len(batch["ids"])
                if id_dict:
                    q_id = id_dict[q_id]
                inputs = {k: v for k, v in batch.items() if k not in {"ids"}}
                for k, v in inputs.items():
                    inputs[k] = v.to(self.device)
                with torch.amp.autocast("cuda", dtype=dtype):
                    query = self.model.encode(**inputs)  # we assume ONE query per batch here
                # Alpha pruning: keep fewest top values whose cumsum >= alpha * L1 sum
                '''
                alpha = 0.65
                
                l1_sum = query.sum(dim=-1, keepdim=True)
                sorted_vals, sorted_idx = torch.sort(query, dim=-1, descending=True)
                cumsum = torch.cumsum(sorted_vals, dim=-1)
                cutoff = (cumsum >= alpha * l1_sum).long().argmax(dim=-1)
                positions = torch.arange(query.size(-1), device=query.device)
                keep_mask = positions.unsqueeze(0) <= cutoff.unsqueeze(1)
                pruned = torch.zeros_like(query)
                pruned.scatter_(-1, sorted_idx, sorted_vals * keep_mask)
                query = pruned
                '''
                if self.compute_stats:
                    stats["L0_q"] += self.l0(query).item()
                # TODO: batched version for retrieval
                row, col = torch.nonzero(query, as_tuple=True)
                values = query[to_list(row), to_list(col)]
                # print("torch col: ", col, col.dtype) 
                # print("torch vals: ", values, values.dtype)
                # print(col.cpu().numpy())
                # print("col: ", col.cpu().numpy().dtype, col.cpu().numpy().shape)
                # print("values: ", values.cpu().numpy().dtype, values.cpu().numpy().shape)
                filtered_indexes, scores = self.numba_score_float(self.numba_index_doc_ids,
                                                                  self.numba_index_doc_values,
                                                                  col.cpu().numpy().astype(np.int32),
                                                                  values.cpu().numpy().astype(np.float32),
                                                                  threshold=threshold,
                                                                  size_collection=self.sparse_index.nb_docs())
                # threshold set to 0 by default, could be better
                filtered_indexes, scores = self.select_topk(filtered_indexes, scores, k=top_k)
                for id_, sc in zip(filtered_indexes, scores):
                    res[q_id][str(self.doc_ids[id_])] = float(sc)
                
        
        # write to disk 
        if self.compute_stats:
            stats = {key: value / len(q_loader) for key, value in stats.items()}
        if self.compute_stats:
            with open(os.path.join(self.out_dir, "stats",
                                   "q_stats{}.json".format("_iter_{}".format(name) if name is not None else "")),
                      "w") as handler:
                json.dump(stats, handler)
            if self.doc_stats is not None:
                with open(os.path.join(self.out_dir, "stats",
                                       "d_stats{}.json".format("_iter_{}".format(name) if name is not None else "")),
                          "w") as handler:
                    json.dump(self.doc_stats, handler)
        with open(os.path.join(self.out_dir, "run{}.json".format("_iter_{}".format(name) if name is not None else "")),
                  "w") as handler:
            json.dump(res, handler)
        
        # return values
        if return_d:
            out = {"retrieval": res}
            if self.compute_stats:
                out["stats"] = stats if self.doc_stats is None else {**stats, **self.doc_stats}
            return out      
        
        
class TermEncoderRetriever:
    def __init__(self, model, args):
        self.model = model
        self.model.eval()
        self.args = args 

    def get_doc_scores(self, pred_scores, doc_encodings):
        """
        Args:
            pred_scores: [bz, vocab_size]
            doc_encodings: [N, L]
        Returns:
            doc_scores: [bz, N]
        """
        doc_scores = []
        for i in range(0, len(doc_encodings), 1_000_000):
            batch_doc_encodings = doc_encodings[i: i+1_000_000]
            batch_doc_scores = pred_scores[:, batch_doc_encodings].sum(dim=-1) #[bz, 1_000_000]
            doc_scores.append(batch_doc_scores)
        doc_scores = torch.hstack(doc_scores)
        # Use advanced indexing to get the values from pred_scores
        #selected_scores = pred_scores[:, doc_encodings]  # shape: [bz, N, L]

        # Sum over the last dimension to get the document scores
        #doc_scores = selected_scores.sum(dim=-1)  # shape: [bz, N]

        return doc_scores


    def retrieve(self, collection_loader, docid_to_smtids, topk, out_dir, use_fp16=False, run_name=None):
        if is_first_worker():
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

        # get doc_encodings
        doc_encodings = []
        docids = []
        for docid, smtids in docid_to_smtids.items():
            assert len(smtids) in {16, 32, 64, 128}, smtids 
            doc_encodings.append(smtids)
            docids.append(docid)
        print("length of doc_encodings = {}, docids = {}".format(len(doc_encodings), len(docids)))
        
        if hasattr(self.model, "base_model"):
            doc_encodings = torch.LongTensor(doc_encodings).to(self.model.base_model.device)
        elif hasattr(self.model, "encoder_decoder"):
            doc_encodings = torch.LongTensor(doc_encodings).to(self.model.encoder_decoder.device)
        else:
            raise NotImplementedError

        qid_to_rankdata = {}
        for i, batch in tqdm(enumerate(collection_loader), disable = not is_first_worker(), 
                                    desc=f"encode # {len(collection_loader)} seqs",
                            total=len(collection_loader)):
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    if hasattr(self.model, "base_model"):
                        inputs = {k:v.to(self.model.base_model.device) for k, v in batch.items() if k != "queries"}
                    elif hasattr(self.model, "encoder_decoder"):
                        inputs = {k:v.to(self.model.encoder_decoder.device) for k, v in batch.items() if k != "queries"}
                    else:
                        raise NotImplementedError
                    batch_preds = self.model.lex_encode(**inputs) #[bz, vocab_size]
                    if isinstance(batch_preds, tuple):
                        assert batch_preds[1] == None, batch_preds
                        assert len(batch_preds) == 2, len(batch_preds)
                        batch_preds = batch_preds[0]
                    elif isinstance(batch_preds, torch.Tensor):
                        pass 
                    else:
                        raise NotImplementedError 
                
                    batch_doc_scores = self.get_doc_scores(batch_preds, doc_encodings)
                    top_scores, top_idxes = torch.topk(batch_doc_scores, k=topk, dim=-1) 
                
            if isinstance(batch["queries"], list):
                query_ids = batch["queries"]
            else:
                raise ValueError("query_ids with type {} is not valid".format(type(query_ids)))
            
            for qid, scores, idxes in zip(query_ids, top_scores, top_idxes):
                qid_to_rankdata[qid] = {}
                scores = scores.cpu().tolist()
                idxes = idxes.cpu().tolist()
                for s, idx in zip(scores, idxes):
                    qid_to_rankdata[qid][docids[idx]] = s 

        if run_name is None:
            with open(os.path.join(out_dir, "run.json"), "w") as fout:
                ujson.dump(qid_to_rankdata, fout)
        else:
            with open(os.path.join(out_dir, run_name), "w") as fout:
                ujson.dump(qid_to_rankdata, fout)
                
                
class HybridIndexer:
    def __init__(self, 
                 model, 
                 sparse_index_dir, 
                 dense_index_dir,
                 device, 
                 chunk_size=2_000_000,
                 compute_stats=False, 
                 dim_voc=None, 
                 force_new=True,
                 filename="array_index.h5py", 
                 **kwargs):
        # handle sparse and dense retrieval simultaneously 
        self.model = model
        self.model.eval()
        self.sparse_index_dir = sparse_index_dir
        self.dense_index_dir = dense_index_dir
        self.chunk_size = chunk_size
        
        self.sparse_index = IndexDictOfArray(self.sparse_index_dir, dim_voc=dim_voc, force_new=force_new, filename=filename)
        self.compute_stats = compute_stats
        self.device = device
        if self.compute_stats:
            self.l0 = L0()
        
        self.model.to(self.device)
        self.local_rank = self.device
        assert self.local_rank == torch.distributed.get_rank(), (self.local_rank, torch.distributed.get_rank())
        self.world_size = torch.distributed.get_world_size()
        print("world_size: {}, local_rank: {}".format(self.world_size, self.local_rank))
        

    def index(self, collection_loader, id_dict=None):
        if supports_bfloat16():
            dtype = torch.bfloat16
            print("Using bfloat16")
        else:
            dtype = torch.float32
            print("Using float32")

        # we change doc_ids from list to dict to support muliti-gpu indexing
        doc_ids = {}
        if self.compute_stats:
            stats = defaultdict(float)
        count = 0
        embeddings = []
        embeddings_ids = []
        chunk_idx = 0
        write_freq = self.chunk_size // collection_loader.batch_size
        with torch.inference_mode():
            for idx, batch in enumerate(tqdm(collection_loader, disable = not is_first_worker())):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in {"ids"}}
                with torch.amp.autocast("cuda", dtype=dtype):
                    batch_sparse_reps, batch_dense_reps = self.model.encode(**inputs) #[bz, vocab_size]
                    
                if isinstance(batch["ids"], torch.Tensor):
                    batch_ids = to_list(batch["ids"])
                else:
                    assert isinstance(batch["ids"], list)
                    batch_ids = batch["ids"]
                if id_dict:
                    batch_ids = [id_dict[x] for x in batch_ids]
                
                if self.compute_stats:
                    stats["L0_d"] += self.l0(batch_sparse_reps).item()
                
                # sparse 
                row, col = torch.nonzero(batch_sparse_reps, as_tuple=True)
                data = batch_sparse_reps[row, col]
                row = row + count
                g_row = row.cpu().numpy() * self.world_size + self.local_rank
                
                unique_g_row = np.sort(np.unique(g_row))
                # we delete the code in hybrid indexer that handles unique_g_row < len(batch_ids)
                # As we believe our sparse reps is good enough, such that every docids has the 
                # posting list. If we face the bug again. Please check the code in SparseIndexer
                assert len(unique_g_row) == len(batch_ids), (len(unique_g_row), len(batch_ids))
                doc_ids.update({int(x): y for x, y in zip(unique_g_row, batch_ids)})   
                self.sparse_index.add_batch_document(g_row, col.cpu().numpy(), data.float().cpu().numpy(), n_docs=len(batch_ids)) 
                count += len(batch_ids)
                
                # dense 
                embeddings.append(batch_dense_reps.float().cpu().numpy())
                assert isinstance(batch_ids, list)
                embeddings_ids.extend(batch_ids)
                
                if (idx + 1) % write_freq == 0:
                    embeddings = np.concatenate(embeddings)
                    embeddings_ids = np.array(embeddings_ids, dtype=np.int64)
                    assert len(embeddings) == len(embeddings_ids), (len(embeddings), len(embeddings_ids))

                    text_path = os.path.join(self.dense_index_dir, "embs_{}_{}.npy".format(self.local_rank, chunk_idx))
                    id_path = os.path.join(self.dense_index_dir, "ids_{}_{}.npy".format(self.local_rank, chunk_idx))
                    np.save(text_path, embeddings)
                    np.save(id_path, embeddings_ids)

                    del embeddings, embeddings_ids
                    embeddings, embeddings_ids = [], []

                    chunk_idx += 1
                    
        # dense 
        if len(embeddings) != 0:
            embeddings = np.concatenate(embeddings)
            embeddings_ids = np.array(embeddings_ids, dtype=np.int64)
            assert len(embeddings) == len(embeddings_ids), (len(embeddings), len(embeddings_ids))
            print("last embedddings shape = {}".format(embeddings.shape))
            text_path = os.path.join(self.dense_index_dir, "embs_{}_{}.npy".format(self.local_rank, chunk_idx))
            id_path = os.path.join(self.dense_index_dir, "ids_{}_{}.npy".format(self.local_rank, chunk_idx))
            np.save(text_path, embeddings)
            np.save(id_path, embeddings_ids)

            del embeddings, embeddings_ids
            chunk_idx += 1 
            
        plan = {"nranks": torch.distributed.get_world_size(), 
                "num_chunks": chunk_idx, 
                "index_path": os.path.join(self.dense_index_dir, "model.index")}
        print("plan: ", plan)
        
        if is_first_worker():
            with open(os.path.join(self.dense_index_dir, "plan.json"), "w") as fout:
                ujson.dump(plan, fout)
                
        # sparse         
        if self.compute_stats:
            stats = {key: value / len(collection_loader) for key, value in stats.items()}
        if self.sparse_index_dir is not None:
            self.sparse_index.save()
            pickle.dump(doc_ids, open(os.path.join(self.sparse_index_dir, "doc_ids.pkl"), "wb"))
            print("done iterating over the corpus...")
            print("index contains {} posting lists".format(len(self.sparse_index)))
            print("index contains {} documents".format(len(doc_ids)))
            if self.compute_stats:
                with open(os.path.join(self.sparse_index_dir, "index_stats.json"), "w") as handler:
                    ujson.dump(stats, handler)
        else:
            # if no index_dir, we do not write the index to disk but return it
            for key in list(self.sparse_index.index_doc_id.keys()):
                # convert to numpy
                self.sparse_index.index_doc_id[key] = np.array(self.sparse_index.index_doc_id[key], dtype=np.int32)
                self.sparse_index.index_doc_value[key] = np.array(self.sparse_index.index_doc_value[key],
                                                                  dtype=np.float32)
            out = {"index": self.sparse_index, "ids_mapping": doc_ids}
            if self.compute_stats:
                out["stats"] = stats
            return out
        

class HybridRetriever:
    @staticmethod
    def select_topk(filtered_indexes, scores, k):
        if len(filtered_indexes) > k:
            sorted_ = np.argpartition(scores, k)[:k]
            filtered_indexes, scores = filtered_indexes[sorted_], -scores[sorted_]
        else:
            scores = -scores
        return filtered_indexes, scores

    @staticmethod
    @numba.njit(nogil=True, parallel=True, cache=True)
    def numba_score_float(inverted_index_ids: numba.typed.Dict,
                          inverted_index_floats: numba.typed.Dict,
                          indexes_to_retrieve: np.ndarray,
                          query_values: np.ndarray,
                          threshold: float,
                          size_collection: int):
        scores = np.zeros(size_collection, dtype=np.float32)  # initialize array with size = size of collection
        n = len(indexes_to_retrieve)
        for _idx in range(n):
            local_idx = indexes_to_retrieve[_idx]  # which posting list to search
            query_float = query_values[_idx]  # what is the value of the query for this posting list
            retrieved_indexes = inverted_index_ids[local_idx]  # get indexes from posting list
            retrieved_floats = inverted_index_floats[local_idx]  # get values from posting list
            for j in numba.prange(len(retrieved_indexes)):
                #print(retrieved_indexes[j], retrieved_floats[j])
                scores[retrieved_indexes[j]] += query_float * retrieved_floats[j]
        filtered_indexes = np.argwhere(scores > threshold)[:, 0]  # ideally we should have a threshold to filter
        # unused documents => this should be tuned, currently it is set to 0
        return filtered_indexes, -scores[filtered_indexes]

    def __init__(self, 
                 model, 
                 sparse_index_dir,
                 dense_index_dir,
                 out_dir, 
                 dim_voc, 
                 device, 
                 **kwargs):
        self.model = model 
        self.model.eval()

        self.sparse_index = IndexDictOfArray(sparse_index_dir, dim_voc=dim_voc)
        self.doc_ids = pickle.load(open(os.path.join(sparse_index_dir, "doc_ids.pkl"), "rb"))
        
        # convert to numba
        self.numba_index_doc_ids = numba.typed.Dict()
        self.numba_index_doc_values = numba.typed.Dict()
        for key, value in self.sparse_index.index_doc_id.items():
            self.numba_index_doc_ids[key] = value
        for key, value in self.sparse_index.index_doc_value.items():
            self.numba_index_doc_values[key] = value
        
        self.sparse_out_dir = os.path.join(out_dir, "sparse")
        self.dense_out_dir = os.path.join(out_dir, "dense")
        if is_first_worker():
            os.makedirs(self.sparse_out_dir, exist_ok=True)
            os.makedirs(self.dense_out_dir, exist_ok=True)
        self.l0 = L0()
        
        # dense 
        self.dense_index = DenseFlatIndexer()
        self.dense_index.init_index(self.model.hidden_size)
        doc_vector_files, doc_id_files = obtain_doc_vec_dir_files(dense_index_dir)
        self._index_encoded_data(doc_vector_files, doc_id_files)

        self.device = device
        self.model.to(device)
        
    def _generate_query_vecs(self, q_loader):
        sparse_query_vecs = [] 
        dense_query_vecs = [] 
        qids = []
        
        with torch.inference_mode():
            for t, batch in enumerate(tqdm(q_loader, total=len(q_loader),
                                      desc="generate query vecs", disable = not is_first_worker())):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in {"ids"}}
                with torch.amp.autocast("cuda", dtype=torch.bfloat16 if supports_bfloat16() else torch.float32):
                    batch_sparse_reps, batch_dense_reps = self.model.encode(**inputs)
                
                qids.extend(batch["ids"] if isinstance(batch["ids"], list) else to_list(batch["ids"]))
                dense_query_vecs.append(batch_dense_reps.cpu().numpy())
                for sparse_rep in batch_sparse_reps:
                    sparse_rep = sparse_rep.unsqueeze(0)
                    row, col = torch.nonzero(sparse_rep, as_tuple=True)
                    assert all(row == 0), row
                    data = sparse_rep[row, col]
                    
                    sparse_query_vecs.append((col.cpu().numpy().astype(np.int32), 
                                              data.cpu().numpy().astype(np.float32)))
        dense_query_vecs = np.concatenate(dense_query_vecs, axis=0)
                    
        assert len(sparse_query_vecs) == len(dense_query_vecs) == len(qids), (len(sparse_query_vecs), len(dense_query_vecs), len(qids))
                    
        return sparse_query_vecs, dense_query_vecs, qids
    
    def _index_encoded_data(self, doc_vec_files, doc_id_files):
        doc_reps = []
        doc_ids = []
        for doc_file, id_file in zip(doc_vec_files, doc_id_files):
            doc_reps.append(np.load(doc_file))
            doc_ids.append(np.load(id_file))
            
        doc_reps = np.concatenate(doc_reps, axis=0)
        doc_ids = np.concatenate(doc_ids).tolist()
        
        assert len(doc_reps) == len(doc_ids), (len(doc_reps), len(doc_ids))
        
        print("size of doc reps to index: ", doc_reps.shape)
        self.dense_index.index_data(doc_reps, doc_ids)
        print("finished indexing")
        
    def _dense_retrieve(self, query_reps, qids, topk=1000):
        res = defaultdict(dict)
        # this is for dense retrieval
        top_doc_ids, top_scores = self.dense_index.search_knn(query_reps, topk)
        assert len(qids) == len(query_reps), (len(qids), len(query_reps))
        for qid, docids, scores in zip(qids, top_doc_ids, top_scores):
            for docid, score in zip(docids, scores):
                res[str(qid)][str(docid)] = float(score)
        
        return res
    
    def _sparse_retrieve(self, sparse_query_vecs, qids, threshold=0., topk=1000):
        res = defaultdict(dict)
        stats = defaultdict(float)
        for qid, (col, values) in tqdm(zip(qids, sparse_query_vecs),
                                       desc="retrieval by inverted index", 
                                       disable = not is_first_worker(), total=len(qids)):
            filtered_indexes, scores = self.numba_score_float(self.numba_index_doc_ids,
                                                              self.numba_index_doc_values,
                                                              col, 
                                                              values,
                                                              threshold=threshold,
                                                              size_collection=self.sparse_index.nb_docs())
            # threshold set to 0 by default, could be better
            filtered_indexes, scores = self.select_topk(filtered_indexes, scores, k=topk)
            for id_, sc in zip(filtered_indexes, scores):
                res[str(qid)][str(self.doc_ids[id_])] = float(sc)
            stats["L0_q"] += len(values) / len(qids)
            
        return res, stats

    def retrieve(self, q_loader, topk, id_dict=False, threshold=0.):
        sparse_query_vecs, dense_query_vecs, qids = self._generate_query_vecs(q_loader)
        sparse_res, sparse_stats = self._sparse_retrieve(sparse_query_vecs, qids, threshold=threshold, topk=topk)
        dense_res = self._dense_retrieve(dense_query_vecs, qids, topk=topk)
        
        # sparse 
        with open(os.path.join(self.sparse_out_dir, "q_stats.json"), "w") as handler:    
            json.dump(sparse_stats, handler)
        with open(os.path.join(self.sparse_out_dir, "run.json"),
                  "w") as handler:
            json.dump(sparse_res, handler)
        
        # dense 
        with open(os.path.join(self.dense_out_dir, "run.json"),
                  "w") as handler:
            json.dump(dense_res, handler)
