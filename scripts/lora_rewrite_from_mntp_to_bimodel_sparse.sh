#!/bin/bash 


python preprocess/lora_rewrite_from_mntp_to_bimodel_sparse.py \
    --input_dir /projects/bcgk/yzound/checkpoint/lion/mntp/ \
    --output_dir /projects/bcgk/yzound/checkpoint/lion/mntp/llama3-1b-msmarco/bimodel_sparse 