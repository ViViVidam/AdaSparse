#!/bin/bash

NGPU=2
torchrun --nproc_per_node=$NGPU -m run_mntp train_configs/mntp/meta_llama3_1b_msmarco.json