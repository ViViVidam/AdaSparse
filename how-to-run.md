# How I modified the code and how to run it
Ziao Wang

## Overview
The original repository is at https://github.com/HansiZeng/scaling-retriever/.

Since my code is modified from it, please take a look at their README.md, including how to setup the environment.

### Hard-coded paths

The shell scripts in the original codebase hard-codes various file paths (corpus path, score paths, output directories, etc), for instance:

```shell
model_name_or_path="/gypsum/work1/xxx/yyy/llm_as_retriever/checkpoints/mntp/llama3-1b-msmarco"
...
output_dir=/gypsum/work1/xxx/yyy/llm_as_retriever/checkpoints/$run_name
```

In order to re-run the experiments, I changed some of the paths (output dirs, etc) to hard-coded to my account on NCSA delta, such as:

```shell
--beir_dataset_dir /work/hdd/bcgk/zwang48/beir_dataset/$subset \
--doc_embed_dir /work/hdd/bcgk/zwang48/sclr_embedding/beir/dense-1B/$subset \
```

To run the code yourself, you need to change these paths.

### Datasets for training and evaluation

As shown in README.md, you can download most of them from [MSMARCO Evaluation and Training Data](https://drive.google.com/drive/folders/1KVbSr7yO6Uig6YEJeSBHgrRMLcEhGOc9?usp=sharing).

I was not aware of that at first so I generated most of the input files myself instead of using theirs, such as:
- preprocess/preprocess-passages-cl.py converts Tevatron's train.jsonl to a format accepted by train_sparse.py
- preprocess/calculate-teacher-score.py raeds Tevatron's train.jsonl, calculates the teacher model scores using MiniLM-L6-v2 and then stores the queries, passages and scores in a format accepted by the scripts to train the CL+KD model (scripts/msmarco/llama_1b_sparse_lora_train_cl-kd.sh)

### Added features

I added some features to their code to support my experiments, in the form of command line arguments, such as:
- `--sparsity_limit 200`, an argument accepted by train_sparse.py and eval_sparse.py that enforces a top-200 hard-limit on sparsity during training or inference

## How to run the experiments

Things to change in the shell scripts if you want to train and evaluate the 8B model:
- file paths (train_configs/mntp/meta_llama3_1b_msmarco.json -> train_configs/mntp/meta_llama3_8b_msmarco.json)
- batch size
- memory allocated
- etc.

### MNTP Pre-training and Enabling Bi-directional attention

`sbatch scripts/run_llama_mntp.sh`

Various settings are in the config file train_configs/mntp/meta_llama3_1b_msmarco.json. According to it, the MNTP model is saved to /projects/bcgk/zwang48/sclr/checkpoints/mntp/llama3-1b-msmarco.

Next, enable bi-directional attention, run
- `bash scripts/lora_rewrite_from_mntp_to_bimodel.sh` to generate the base model for dense retrieval models. It is saved to `/projects/bcgk/zwang48/sclr/checkpoints/mntp/llama3-1b-msmarco/bimodel`
- `bash scripts/lora_rewrite_from_mntp_to_bimodel_sparse.sh` to generate the base model for sparse retrieval models. 
It is saved to `/projects/bcgk/zwang48/sclr/checkpoints/mntp/llama3-1b-msmarco/bimodel_sparse`

### Fine-tuning sparse retrieval models

Common Added Optional Arguments involved:
- `--sparsity_limit 200`, limit sparsity during the fine-tuning phase.

#### Contrastive loss only

`sbatch scripts/msmarco/llama_1b_sparse_lora_train_cl.sh`

#### Contrastive loss + Knowledge Distillation loss

`sbatch scripts/msmarco/llama_1b_sparse_lora_train_cl-kd.sh`

### Evaluating the sparse models

There are two steps to evaluation:
1. The codebase first creates an index storing the vector representations of the documents (shell scripts with "index")
2. Then, the codebase loads the index, encodes the query vectors and performs brute-force retrieval (shell scrips with "eval").

Before running step 2, we need to check the index directory to make sure the index is successfully created.

The evaluation metrics are stored in perf.json in the output directory. Evaluation on MS MARCO gives you the performance on MS MARCO Dev and TREC DL 19&20. For BEIR, you can use `python analysis/beir_results.py --base_dir <base_dir>` to find the average performance numbers among the public subsets of BEIR.

The index created by the sparse retriever is in h5py format. In the index directory, you can view the sparsity (L0_d) of the vector representations of the corpus in index_stats.json.

Another thing to note is that we probably want to evaluate with the same added arguments as in fine-tuning. For instance, if the sparse model is fine-tuned using `--sparsity_limit 200`, we probably want to add it in `index_sparse.sh` and `eval_sparse.sh` as well.

#### On MS MARCO
`sbatch scripts/index_sparse.sh` and then `sbatch scripts/eval_sparse.sh`.

#### On BEIR
`sbatch scripts/index_sparse_beir.sh` and then `sbatch scripts/eval_sparse_beir.sh`.

## Experiment Artifacts

### About the output/ folder

I logged the stdout and stderr of the SLURM runs in the output/ folder.

### Model checkpoints

They are stored in `/work/hdd/bcgk/zwang48/model_sclr_ckpts/`. The subdirectories include
- "300steps" "600steps": debug runs where the total number of steps is shortened to 300 or 600 steps.
- "llama3-1b-msmarco": the base models (after MNTP pre-training but before fine-tuning)
- "main": where the model checkpoints reported on my defence are stored.

### Indexes and Retreival Results

They are stored in `/work/hdd/bcgk/zwang48/sclr_embedding/new`, organized by the model name.

Each model's directory may contain the following subdirectories:
- "msmarco": the index of MS MARCO passages
- "msmarco-results": retrieval results on MS MARCO Dev and TREC DL 19&20
- "beir": the indexes of the subsets of BEIR
- "beir-results": retrieval results on BEIR
- "encode-q": the query vectors stored

Because of the limited space of NCSA Delta and the large disk usage of indexes, I have migrated some of the artifacts to Expanse, at `/expanse/lustre/projects/csb176/zwang48/sclr/delta_backup`.

## Appendix

### Tips for using SLURM

To view the status of a previous job, do
`sacct -j <job_num>`
Note that even when the task is actually completed (e.g. 9926630, 9928211) exitcode might also shows -1 because you did not clean up.

To automatically start a job2 after job1 finish,
```
[user@gl-login1]$ sbatch --dependency=afterany:13205 Job2.sh
13206
```