#!/bin/bash

# Unlimited sparse vector requires 128G for MS MARCO, otherwise 96G is enough

model_name_or_path=/projects/bfqn/yzound/checkpoint/lion/llama3-1b-adasparse-test

# eval_batch_size 64, because changed from A100 to A40
python -m scaling_retriever.utils.inverted_index \
	--model_name_or_path $model_name_or_path \
	--index_dir "/work/nvme/bcgk/yzound/index/msmarco/AdaSparse/index"