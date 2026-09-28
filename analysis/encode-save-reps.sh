#!/bin/bash

#/work/hdd/bcgk/zwang48/model_sclr_ckpts/multickpt/llama3-1b-marco-mntp-sparse-nolimit-nce-dfflops3-600steps-x0.3-A1-BlogE6-lora-1e-4_qreg_05_dreg_04_bs_10_epochs_1_nnegs_16/checkpoint-600
#/projects/bcgk/zwang48/sclr/checkpoints/llama3-1b-marco-mntp-sparse-nce-lora-1e-4_qreg_05_dreg_04_bs_10_epochs_1_nnegs_16
# 9906768, 9907011
torchrun --nproc_per_node=1 --master_port 4437 -m analysis.encode-save-reps.py \
  --model_name_or_path /work/hdd/bcgk/zwang48/model_sclr_ckpts/llama3-1b-marco-mntp-sparse-nolimit-nce-dfflops2-C100-lora-1e-4_qreg_05_dreg_04_bs_10_epochs_1_nnegs_16 \
  --save_reps_to /work/hdd/bcgk/zwang48/sclr_embedding/new/sparse-nce-dfflops3-C100/scifact_reps.pkl