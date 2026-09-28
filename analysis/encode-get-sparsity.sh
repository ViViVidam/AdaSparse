#!/bin/bash

#/work/hdd/bcgk/zwang48/model_sclr_ckpts/multickpt/llama3-1b-marco-mntp-sparse-nolimit-nce-dfflops3-600steps-x0.3-A1-BlogE6-lora-1e-4_qreg_05_dreg_04_bs_10_epochs_1_nnegs_16/checkpoint-600
#/projects/bcgk/zwang48/sclr/checkpoints/llama3-1b-marco-mntp-sparse-nce-lora-1e-4_qreg_05_dreg_04_bs_10_epochs_1_nnegs_16
# 9906768, 9907011
torchrun --nproc_per_node=1 --master_port 4473 -m analysis.encode-get-sparsity.py \
  --subset scifact \
  --model_name_or_path /work/hdd/bcgk/zwang48/model_sclr_ckpts/multickpt/llama3-1b-marco-mntp-sparse-nolimit-nce-idfscale-d-lora-1e-4_qreg_05_dreg_04_bs_10_epochs_1_nnegs_16