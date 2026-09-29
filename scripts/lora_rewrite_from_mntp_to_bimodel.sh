#!/bin/bash

# MNTP checkpoint produced by scripts/run_llama_mntp.sh
input_dir="path to MNTP checkpoint directory"
# Where the bi-directional (dense) model is written.
output_dir="path to bimodel output directory"

python preprocess/lora_rewrite_from_mntp_to_bimodel.py \
    --input_dir "$input_dir" \
    --output_dir "$output_dir"
