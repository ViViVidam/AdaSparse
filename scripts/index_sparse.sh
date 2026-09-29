#!/bin/bash

# Unlimited sparse vector requires 128G for MS MARCO, otherwise 96G is enough

NGPU=2

corpus_path="path to collection.tsv"
# CHANGE HERE
model_name_or_path=Johonson/adasparse-1B
index_dir="path to index output directory"

# alpha pruning: keep fewest top values whose cumsum >= alpha * L1 sum; leave unset to disable
alpha=1 #0.65

echo $model_name_or_path
echo "$index_dir"

torchrun --nproc_per_node=$NGPU --master_port 4434 -m eval_sparse \
        --model_name_or_path $model_name_or_path \
        --index_dir "$index_dir" \
        --task_name indexing \
        --eval_batch_size 64 \
        --corpus_path "$corpus_path" \
        --alpha $alpha
        #--out_ciff
if [[ $NGPU > 1 ]]; then
    python -m scaling_retriever.utils.inverted_index \
        --model_name_or_path $model_name_or_path \
        --index_dir "$index_dir"
    else
        :
    fi
