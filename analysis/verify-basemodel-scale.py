"""
Load model checkpoint and check whether it contains the inverse document frequencies used to scale the reps
"""

import os
import sys
sys.path.append(os.getcwd())

from scaling_retriever.modeling.llm_encoder import T5Sparse, LlamaBiSparse
model_name_or_path = "/work/hdd/bcgk/zwang48/model_sclr_ckpts/multickpt/llama3-1b-marco-mntp-sparse-nolimit-nce-idfscale-d-lora-1e-4_qreg_05_dreg_04_bs_10_epochs_1_nnegs_16"
model = LlamaBiSparse.load_from_lora(model_name_or_path)
if hasattr(model.base_model, "doc_reps_scale"):
    print("document reps are scaled")
else:
    print("document reps are NOT scaled")