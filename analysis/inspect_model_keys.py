import os
import sys
sys.path.append(os.getcwd())
"""
import torch
from scaling_retriever.modeling.bidirectional_llama import LlamaBiForMNTP, LlamaBiModel
base_model = LlamaBiForMNTP.from_pretrained('/projects/bcgk/zwang48/sclr/checkpoints/mntp/llama3-1b-msmarco/bimodel_sparse')
print(base_model.model.state_dict().keys())
print("===")
print(base_model.lm_head.state_dict().keys())
lm_head_ckpt = base_model.lm_head.state_dict()
for k,v in lm_head_ckpt.items():
    print(k)
    print(v.shape)
exit(0)
"""

import torch
from safetensors.torch import load_file
# checkpoint = load_file('/projects/bcgk/zwang48/sclr/checkpoints/mntp/llama3-1b-msmarco/bimodel_sparse/adapter_model.safetensors')
# checkpoint = torch.load('/work/hdd/bcgk/zwang48/model_sclr_ckpts/multickpt/llama3-1b-marco-mntp-sparse-nolimit-nce-dfflops3-x0.5-A1-BlogE6-lora-1e-4_qreg_05_dreg_04_bs_10_epochs_1_nnegs_16/adapter_model.bin')
checkpoint = torch.load('/work/hdd/bcgk/zwang48/model_sclr_ckpts/multickpt/llama3-1b-marco-mntp-sparse-200-debug-nce-lora-lmhead1-1e-4_qreg_05_dreg_04_bs_10_epochs_1_nnegs_16/checkpoint-950/adapter_model.bin')
tot_tunable = 0
for k,v in checkpoint.items():
    print(k)
    print(v.shape)
    tot_tunable += torch.prod(torch.tensor(v.shape)).item()
print("total tunable params:", tot_tunable)