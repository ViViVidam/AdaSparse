#!/bin/bash

# Unlimited sparse vector requires 128G for MS MARCO, otherwise 96G is enough

NGPU=2
corpus_path=/projects/bcgk/zwang48/sclr/msmarco-full/collection.tsv
model_name_or_path=/projects/bfqn/yzound/checkpoint/lion/llama3-1b-ce-kldiv-learned-top-6-5k-dq-lora-1e-4_qreg_05_dreg_04_bs_28_epochs_1_nnegs_16_dual_perterm/

# eval_batch_size 64, because changed from A100 to A40
# "arguana" "fiqa" "nfcorpus" "quora" "scidocs" "scifact" "trec-covid" "webis-touche2020"
for subset in "webis-touche2020"; do 
    index_dir=/projects/bgca/yzound/index/beir/lion/lion_1B_6_5/$subset/index
    python -m scaling_retriever.utils.inverted_index \
	    --model_name_or_path $model_name_or_path \
	    --index_dir $index_dir
done