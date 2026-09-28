"""
Analyze the sparsity and FLOPS loss of the of passages.
Since creating, saving and loading the index from disk might take very long time, we encode the passages on the fly.
"""


import os
import sys
sys.path.append(os.getcwd())
import argparse

# copy imports
import torch.distributed
import ujson 
from dataclasses import field, dataclass

import transformers
from transformers import AutoTokenizer, HfArgumentParser
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd 
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from huggingface_hub import hf_hub_download

from scaling_retriever.dataset.dataset import CollectionDataset, WikiQueryDataset, MSMARCOQueryDataset, BeirDataset
from scaling_retriever.dataset.data_collator import T5SparseCollectionCollator, LlamaSparseCollectionCollator
from scaling_retriever.modeling.llm_encoder import T5Sparse, LlamaBiSparse
from scaling_retriever.modeling.losses.regulariaztion import L0, FLOPS 
from scaling_retriever.utils.utils import supports_bfloat16
from scaling_retriever.indexer import SparseIndexer, SparseRetrieval
import constants
from scaling_retriever.utils.metrics import load_and_evaluate, evaluate_beir

parser = argparse.ArgumentParser("use model to encode the corpus to get its sparsity and flops loss, does not save vectors to index")
parser.add_argument('--model_name_or_path', type=str)
parser.add_argument('--subset', type=str, default="scifact")
parser.add_argument('--sparsity_limit', type=int, default=None)
args = parser.parse_args()

init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])

subset=args.subset
print("subset: ", subset)
if subset == "msmarco":
    corpus_path="/projects/bcgk/zwang48/sclr/msmarco-full/collection.tsv"
    d_collection = CollectionDataset(corpus_path=corpus_path, data_source=constants.corpus_datasource[corpus_path])
    # cannot encode all of msmarco, so sample
    d_collection.pids = d_collection.pids[:10000]
else: # BEIR
    beir_dataset_dir = f"/work/hdd/bcgk/zwang48/beir_dataset/{subset}"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{subset}.zip"
    data_path = util.download_and_unzip(url, beir_dataset_dir)
    corpus, _, _ = GenericDataLoader(data_folder=data_path).load(split="test")
    d_collection = BeirDataset(corpus, information_type="document")
print(f"num passages: {len(d_collection)}")

model_name_or_path=args.model_name_or_path
print(f"model_name_or_path: {model_name_or_path}")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
except: # we are in a numbered checkpoint
    tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(model_name_or_path))
model = LlamaBiSparse.load_from_lora(model_name_or_path).to('cuda')
if hasattr(args, "sparsity_limit") and args.sparsity_limit != None:
    model.sparsity_limit = args.sparsity_limit
d_collator = LlamaSparseCollectionCollator(tokenizer=tokenizer, max_length=512)
d_loader = DataLoader(d_collection, batch_size=48, shuffle=False, 
                    collate_fn=d_collator, num_workers=2,
                    sampler=DistributedSampler(d_collection, shuffle=False))

# Vocabulary is 128002, but in GPU tensor shape is 128256 to be a multiple of 256 
sum_pool = torch.zeros(128256).to('cuda')
L0_d = 0
with torch.inference_mode():
    for t, batch in enumerate(tqdm(d_loader)):
        inputs = {k: v.to('cuda') for k, v in batch.items() if k not in {"ids"}}
        with torch.amp.autocast("cuda", dtype=torch.float32):
            batch_documents = model.encode(**inputs) #[bz, vocab_size]
            L0_d += torch.count_nonzero(batch_documents).item()
            batch_sum = torch.sum(batch_documents,0)
            sum_pool += batch_sum
sum_pool = sum_pool.cpu()
sum_pool /= len(d_collection)
L0_d /= len(d_collection)

print("L0_d")
print(L0_d)
print("Sum of positions")
print(torch.sum(sum_pool).item())
print("Sum of positions squared")
print(torch.sum(sum_pool ** 2).item())