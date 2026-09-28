import os 
import wandb
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, Union
import logging
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer,
                          BertForSequenceClassification, 
                          BertTokenizer,
                          HfArgumentParser,
                          TrainingArguments,
                          Trainer)
import transformers
from transformers.modeling_utils import unwrap_model
from transformers.trainer_utils import EvalPrediction
from torch.utils.data.dataloader import DataLoader
import ujson
import torch
from copy import deepcopy
import numpy as np
import json

TRAINING_ARGS_NAME = "training_args.bin"

from scaling_retriever.dataset.dataset import (
    DualEncoderDatasetForNCE, 
    DualEncoderDatasetForMarginMSE,
    DualEncoderDatasetForKLDiv
)
from scaling_retriever.dataset.data_collator import (
    LlamaSparseCollatorForNCE,
    LlamaSparseCollatorForMarginMSE,
    LlamaSparseCollatorForNCE_KLDiv,
    LlamaSparseCollatorForKLDiv,
)
from scaling_retriever.modeling.llm_encoder import (
    LlamaBiSparse,
    LlamaBiSparseForMarginMSE,
    LlamaBiSparseForNCE_KLDiv_AdaSparse,
    LlamaBiSparseForNCE_KLDiv,
    LlamaBiSparseForKLDiv,
)
from scaling_retriever.tasks.sparse_trainer import LLM2RetrieverTrainingArgs, SparseTrainer
from scaling_retriever.modeling.losses.regulariaztion import RegWeightScheduler
from scaling_retriever.utils.utils import get_data_source

logger = logging.getLogger(__name__)

def save_training_args(model_args, args, output_dir):
    merged_args = {**asdict(model_args), **asdict(args)}
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "args.json"), "w") as fout:
        ujson.dump(merged_args, fout, indent=4)
        
def get_model_type(model_name_or_path):
    config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    print("model_type: ", config.model_type) 
    return config.model_type

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def load_from_adpater(args, model_cls,load_threshold=False,resize_len=None,thresh_size=1):
    with open(os.path.join(args.model_name_or_path, "adapter_config.json"), "r") as f:
        adapter_config = ujson.load(f)
    base_model_name_or_path = adapter_config["base_model_name_or_path"]
    print("load lora model from ", args.model_name_or_path)
    lora_lm_head_r = args.lora_lm_head_r if (hasattr(args,"lora_lm_head_r") and args.lora_lm_head_r!=None) else None
    model = model_cls.load(base_model_name_or_path,
                            lora_name_or_path=args.model_name_or_path,
                            merge_peft=False,
                            is_trainable=True,
                            lora_lm_head_r=lora_lm_head_r,
                            load_threshold=load_threshold,resize_len=resize_len,thresh_size=thresh_size)
    return model
def sub_processing(model):
    if hasattr(args, "sparsity_limit") and args.sparsity_limit != None:
        model.sparsity_limit = args.sparsity_limit
if __name__ == "__main__":
    parser = HfArgumentParser((LLM2RetrieverTrainingArgs))
    args = parser.parse_args_into_dataclasses()[0]
    if args.local_rank <= 0:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w") as fout:
            ujson.dump(asdict(args), fout, indent=4)
            
    # Identify the model_type and either dense and sparse model 
    if args.model_type is None:
        model_type = get_model_type(args.model_name_or_path)
    else:
        model_type = args.model_type
    
    if args.loss_type == "nce":
        train_dataset= DualEncoderDatasetForNCE(
            corpus_path=args.corpus_path,
            train_path=args.train_path,
            data_source=get_data_source(args),
            n_negs=args.n_negs
        )
    elif args.loss_type == "margin_mse":
        train_dataset = DualEncoderDatasetForMarginMSE(
            corpus_path=args.corpus_path,
            train_path=args.train_path,
            data_source=get_data_source(args),
        )
    elif args.loss_type in ["nce_kldiv", "kldiv","adasparse"]:
        train_dataset = DualEncoderDatasetForKLDiv(
            corpus_path=args.corpus_path,
            train_path=args.train_path,
            data_source=get_data_source(args),
            n_negs=args.n_negs
        )
    
    if os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json")):
        # It is hacky here and inconsistent with train_splade.py
        # As we transform the lora_model's state_dict and config from MNTP to BiModel
        # in the new folder without copy the tokenizer configs.
        # Hence we will load the tokenizer from the base_model_name_or_path
        with open(os.path.join(args.model_name_or_path, "adapter_config.json"), "r") as f:
            adapter_config = ujson.load(f)
        tokenizer = AutoTokenizer.from_pretrained(adapter_config["base_model_name_or_path"])
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    if model_type == "llama":
        if args.train_config is not None:
            with open(args.train_config, "r") as fin:
                config = ujson.load(fin)
        else:
            config = None
        if args.loss_type == "nce":
            train_collator = LlamaSparseCollatorForNCE(tokenizer=tokenizer, query_max_length=args.query_max_length,
                                                        doc_max_length=args.doc_max_length)
            if os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json")):
                model = load_from_adpater(args, LlamaBiSparse)
            else:
                model = LlamaBiSparse.build(args.model_name_or_path, args, config=config)
        elif args.loss_type == "margin_mse":
            train_collator = LlamaSparseCollatorForMarginMSE(tokenizer=tokenizer, query_max_length=args.query_max_length,
                                                        doc_max_length=args.doc_max_length)
            if os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json")):
                model = load_from_adpater(args, LlamaBiSparseForMarginMSE)
            else:
                model = LlamaBiSparseForMarginMSE.build(args.model_name_or_path, args, config=config)
        elif args.loss_type == "adasparse":
            train_collator = LlamaSparseCollatorForNCE_KLDiv(tokenizer=tokenizer, query_max_length=args.query_max_length,
                                                        doc_max_length=args.doc_max_length)
            if os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json")):
                # per-term thresholding: one learnable threshold per vocabulary term
                thresh_size = len(tokenizer)
                model = load_from_adpater(args, LlamaBiSparseForNCE_KLDiv_AdaSparse,load_threshold=args.thresh,resize_len=len(tokenizer),thresh_size=thresh_size)
            else:
                model = LlamaBiSparseForNCE_KLDiv_AdaSparse.build(args.model_name_or_path, args, config=config,
                                                                  thresh_size=len(tokenizer))
                print("initialized from class LlamaBiSparseForNCE_KLDiv_AdaSparse")
        elif args.loss_type == "nce_kldiv":
            train_collator = LlamaSparseCollatorForNCE_KLDiv(tokenizer=tokenizer, query_max_length=args.query_max_length,
                                                        doc_max_length=args.doc_max_length)
            if os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json")):
                model = load_from_adpater(args, LlamaBiSparseForNCE_KLDiv)
            else:
                model = LlamaBiSparseForNCE_KLDiv.build(args.model_name_or_path, args, config=config)
        elif args.loss_type == "kldiv":
            train_collator = LlamaSparseCollatorForKLDiv(tokenizer=tokenizer, query_max_length=args.query_max_length,
                                                        doc_max_length=args.doc_max_length)
            if os.path.exists(os.path.join(args.model_name_or_path, "adapter_config.json")):
                model = load_from_adpater(args, LlamaBiSparseForKLDiv)
            else:
                model = LlamaBiSparseForKLDiv.build(args.model_name_or_path, args, config=config)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        if args.use_bow:
            model.base_model.config.update({"use_bow":True})
        else:
            model.base_model.config.update({"use_bow":False})
        model.base_model.config.update({"bow_doc_scale": args.bow_doc_scale})
        model.base_model.config.update({"bow_query_scale": args.bow_query_scale})
        model.base_model.config.update({"lexical_preserve": args.lexical_preserve})


    sub_processing(model)

    model.train()

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    training_args = args
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    
    reg_to_reg_scheduler = {"doc_reg": RegWeightScheduler(lambda_=training_args.ln_to_weight["doc_reg"], 
                                                          T=training_args.max_steps // 3),
                            "query_reg": RegWeightScheduler(lambda_=training_args.ln_to_weight["query_reg"], 
                                                            T=training_args.max_steps // 3)}
    #(model.base_model.config,flush=True)
    # print(args)
    trainer = SparseTrainer(
        model=model,
        tokenizer=tokenizer, 
        train_dataset=train_dataset,
        data_collator=train_collator,
        args=training_args,
        reg_to_reg_scheduler=reg_to_reg_scheduler
    )
    
    if trainer.args.local_rank <= 0:  # only on main process
        wandb.login()
        wandb.init(project=args.wandb_project_name, name=args.run_name)

        # let's save tokenizer first 
        tokenizer.save_pretrained(args.output_dir)

        print("lambda for doc_reg = {}, for query_reg = {}, T = {}".format(
            reg_to_reg_scheduler["doc_reg"].lambda_, reg_to_reg_scheduler["query_reg"].lambda_, reg_to_reg_scheduler["doc_reg"].T
        ))
        print("model: ", args.model_name_or_path, model_type)
        if hasattr(model,"sparsity_limit"):
            print(f"sparse limit set to {model.sparsity_limit}")

        print("----------------------------------------")
        print(f"all of following will be trained and save:")
        model.train()
        for name,params in model.named_parameters():
            if params.requires_grad == True:
                print(name)
        print("----------------------------------------")
        #print(model.base_model.config)
    
    trainer.train()
    if trainer.is_fsdp_enabled:
        trainer.save_model()
    else:
        if trainer.args.local_rank <= 0:
            trainer.save_model()
            