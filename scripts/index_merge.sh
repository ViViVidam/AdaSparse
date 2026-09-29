#!/bin/bash

# Unlimited sparse vector requires 128G for MS MARCO, otherwise 96G is enough

# CHANGE HERE
model_name_or_path=Johonson/adasparse-1B
# index directory written by scripts/index_sparse.sh
index_dir="path to index directory"

# eval_batch_size 64, because changed from A100 to A40
python -m scaling_retriever.utils.inverted_index \
	--model_name_or_path $model_name_or_path \
	--index_dir "$index_dir"
