#!/bin/bash

# MNTP checkpoint produced by scripts/run_llama_mntp.sh
input_dir="path to MNTP checkpoint directory"
# Where the bi-directional sparse model is written.
# Use this directory as model_name_or_path in the fine-tuning scripts.
output_dir="path to bimodel_sparse output directory"

python preprocess/lora_rewrite_from_mntp_to_bimodel_sparse.py \
    --input_dir "$input_dir" \
    --output_dir "$output_dir"
