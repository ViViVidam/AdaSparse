#!/bin/bash 


python preprocess/lora_rewrite_from_mntp_to_bimodel.py \
    --input_dir /projects/bcgk/zwang48/sclr/checkpoints/mntp/llama3-1b-msmarco \
    --output_dir /projects/bcgk/zwang48/sclr/checkpoints/mntp/llama3-1b-msmarco/bimodel 