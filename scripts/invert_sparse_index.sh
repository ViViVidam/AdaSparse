#!/bin/bash

# CHANGE HERE
model_name_or_path=Johonson/adasparse-1B
index_dir="path to index directory"
index_name=sparse

python -m scaling_retriever.utils.inverted_index \
    --model_name_or_path "$model_name_or_path" \
    --index_dir "$index_dir" \
    --index_name "$index_name"
