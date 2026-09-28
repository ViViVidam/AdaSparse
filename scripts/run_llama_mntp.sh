#!/bin/bash

NGPU=2
torchrun --nproc_per_node=$NGPU -m run_mntp train_configs/mntp/meta_llama2_7b_msmarco.json