# Adaptive Sparsity Optimization with Learnable Soft Top-K and Per-Term Thresholding for Efficient Retrieval

[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3805712.3809625-b31b1b.svg)](https://doi.org/10.1145/3805712.3809625)
[![HF Link](https://img.shields.io/badge/HF%20Models-AdaSparse-FFD21E.svg)](https://huggingface.co/Johonson/adasparse-1B)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ViViVidam/AdaSparse/blob/main/LICENSE)

This repository is the official implementation of **AdaSparse** (SIGIR 2026).

AdaSparse ("Adaptive Sparsification") is a learned sparse retriever built on LLaMA backbones. On top of contrastive + knowledge-distillation training, it optimizes model sparsity through a synergy of adaptive strategies: **STop**, a learnable soft top-K regularization that adapts the number of active terms to each query/document, and **PTT**, per-term thresholding that prunes low-weight terms with individualized thresholds, working in complement with FLOPs regularization.

This repository contains the full pipeline: MNTP pre-training, bi-directional conversion, fine-tuning, indexing, and evaluation on MS MARCO / TREC DL / BEIR. The codebase is derived from [scaling-retriever](https://github.com/HansiZeng/scaling-retriever) (Lion).

## Released Model Checkpoints

| Model | Base | HF Hub |
|---|---|---|
| AdaSparse-1B | meta-llama/Llama-3.2-1B | [Johonson/adasparse-1B](https://huggingface.co/Johonson/adasparse-1B) |
| AdaSparse-8B | meta-llama/Meta-Llama-3-8B | [Johonson/adasparse-8B](https://huggingface.co/Johonson/adasparse-8B) |

The base models are gated on HuggingFace — request access on their model pages first.

## Environment Setup

```bash
pip install -r requirements.txt
conda install -c pytorch faiss-cpu=1.8.0
```

Every `.sh` file in this repository is a plain bash script. Run them with `bash` from the
repository root, with your Python environment already active and a working CUDA
toolchain on the path. They do not activate an environment or request resources
for you, so adapt them to your own scheduler if you submit them as batch jobs.

## Quick Start

```python
import torch
from transformers import AutoTokenizer
from scaling_retriever.modeling.llm_encoder import LlamaBiSparse

model = LlamaBiSparse.load_from_lora("Johonson/adasparse-1B")
tokenizer = AutoTokenizer.from_pretrained("Johonson/adasparse-1B")

queries = ["What is the capital of France?", "Who wrote '1984'?"]
passages = [
    "Paris is the capital of France.",
    "George Orwell wrote '1984'."
]
tokenized_queries = tokenizer(queries, max_length=192, truncation=True,
                              padding="longest", return_tensors="pt")
tokenized_passages = tokenizer(passages, max_length=192, truncation=True,
                               padding="longest", return_tensors="pt")

query_embeds = model.query_encode(**tokenized_queries)
doc_embeds = model.doc_encode(**tokenized_passages)

scores = torch.matmul(query_embeds, doc_embeds.T)
print(scores.tolist())
```

## Data

Training and evaluation data for MS MARCO can be downloaded from the
[MSMARCO Evaluation and Training Data](https://drive.google.com/drive/folders/1IkWi7ZB7iRuUuzmwS1tOX9wOCWy8KdnA?usp=drive_link) folder.
BEIR subsets are downloaded automatically at indexing time.

Each training script begins with `corpus_path`, `train_path`, and `model_name_or_path` set to
placeholder strings — fill these in before running. Checkpoints are written to `./checkpoints/<run_name>`.
The indexing and evaluation scripts still carry example index and output directories, so edit those
to match your setup. Retrieval on custom query/corpus files also requires registering the paths in `constants.py`.

## Training

The full pipeline from a vanilla LLaMA checkpoint:

1. **MNTP pre-training**: `bash scripts/run_llama_mntp.sh` (configs in `train_configs/mntp/`)
2. **Enable bi-directional attention**: `bash scripts/lora_rewrite_from_mntp_to_bimodel_sparse.sh`
3. **AdaSparse fine-tuning (CL + KD)**:

```bash
bash scripts/msmarco/llama_1b_adasparse_cl-kd.sh   # 1B
bash scripts/msmarco/llama_8b_adasparse_cl-kd.sh   # 8B
```

Key AdaSparse arguments (`--loss_type=adasparse`):
- `--use_bow`: enable STop, the learnable soft top-K regularization (`L_top` in the paper)
- `--bow_doc_scale` / `--bow_query_scale`: the soft expansion factor `b` that sets the soft limit `K = b · |s|` (paper defaults: `b=8` for documents, `b=10` for queries)
- `--lexical_preserve`: the lexical provenance bias `β` that exempts original-text terms from the gating penalty (paper default: `β=2`)
- `--thresh`: enable PTT, per-term thresholding — one learnable threshold per vocabulary term (`τ_j` in the paper), saved with the adapter

Baseline loss types are also supported: `nce` (CL), `margin_mse` (KD), `nce_kldiv` (CL+KD), via `scripts/msmarco/llama_{1b,3b,8b}_sparse_lora_train_{cl,kd,cl-kd}.sh`.

## Indexing and Evaluation

Evaluation is two steps: build the index, then retrieve.

### MS MARCO

```bash
bash scripts/index_sparse.sh   # multi-GPU indexing; merges shards afterwards
bash scripts/eval_sparse.sh    # MS MARCO Dev + TREC DL 19/20, metrics in perf.json
```

Useful indexing options (see `eval_sparse.py`):
- `--alpha 0.65`: mass ratio pruning (MRP, one of the baselines compared in the paper) applied as post-processing — per vector, keep the fewest top terms whose cumulative weight reaches fraction `α` of the L1 mass (`--alpha 1` disables pruning)
- `--quant`: 8-bit impact quantization (weights scaled ×100, clamped to [0, 255])
- `--out_ciff`: additionally dump the collection as `index.jsonl.gz` (and queries as `queries.jsonl` via `scripts/encode_sparse_query.sh`) for CIFF/impact-index toolchains
- `--ciff_term_ids`: in the ciff output, write token ids as terms instead of the token strings

With multiple GPUs, each rank writes an index shard (`index_0`, `index_1`, ...); `python -m scaling_retriever.utils.inverted_index` merges them, verifies the global non-zero entry counts against the per-rank `index_stats.json`, and removes the shards on success.

### BEIR

```bash
bash scripts/index_sparse_beir.sh
bash scripts/eval_sparse_beir.sh
python analysis/beir_results.py --base_dir <base_dir>   # average over subsets
```

### ⚠ CPU usage for retrieval

Retrieval uses multi-threaded traversal of the inverted index — use **more than 32 CPUs** or it will be significantly slower. On MS MARCO Dev, retrieval typically completes in ~15 minutes under good CPU conditions.

## Citation

```bibtex
@inproceedings{xie2026adasparse,
  title     = {Adaptive Sparsity Optimization with Learnable Soft Top-K and Per-Term Thresholding for Efficient Retrieval},
  author    = {Xie, Wentai and Carlson, Parker and He, Shanxiu and Yang, Tao},
  booktitle = {Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '26)},
  year      = {2026},
  doi       = {10.1145/3805712.3809625}
}
```
