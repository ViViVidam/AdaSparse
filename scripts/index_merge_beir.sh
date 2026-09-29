#!/bin/bash

# Unlimited sparse vector requires 128G for MS MARCO, otherwise 96G is enough

NGPU=2
corpus_path="path to collection.tsv"
# CHANGE HERE
model_name_or_path=Johonson/adasparse-1B
# per-subset index root written by scripts/index_sparse_beir.sh
index_path="path to BEIR index directory"

# eval_batch_size 64, because changed from A100 to A40
# "arguana" "fiqa" "nfcorpus" "quora" "scidocs" "scifact" "trec-covid" "webis-touche2020"
for subset in "webis-touche2020"; do
    index_dir="$index_path/$subset/index"
    python -m scaling_retriever.utils.inverted_index \
	    --model_name_or_path $model_name_or_path \
	    --index_dir "$index_dir"
done
