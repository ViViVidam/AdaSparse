#!/bin/bash

model_name_or_path=/projects/bcgk/zwang48/sclr/checkpoints/llama3-1b-marco-mntp-sparse-nce-lora-1e-4_qreg_05_dreg_04_bs_10_epochs_1_nnegs_16
python -m scaling_retriever.utils.inverted_index --model_name_or_path $model_name_or_path --index_dir /work/hdd/bcgk/zwang48/sclr_embedding/sparse-1B --index_name sparse