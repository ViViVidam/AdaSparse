#!/bin/bash

# Unlimited sparse vector requires 128G for MS MARCO, otherwise 96G is enough

NGPU=2

#corpus_path=/projects/bcgk/yzound/sclr/msmarco-full/sampled_collection.tsv
#model_name_or_path=/projects/bfqn/yzound/checkpoint/lion/llama3-1b-ce-kldiv-learned-top-6-5k-dq-lora-1e-4_qreg_05_dreg_04_bs_28_epochs_1_nnegs_16_dual_perterm/
#index_dir=/work/nvme/bcgk/yzound/index/msmarco/sampled/index

corpus_path=/projects/bcgk/zwang48/sclr/msmarco-full/collection.tsv
model_name_or_path=Johonson/adasparse-1B
index_dir=/work/nvme/bcgk/yzound/index/msmarco/adasparse/index

# alpha pruning: keep fewest top values whose cumsum >= alpha * L1 sum; leave unset to disable
alpha=1 #0.65

echo $model_name_or_path
echo $index_dir

torchrun --nproc_per_node=$NGPU --master_port 4434 -m eval_sparse \
        --model_name_or_path $model_name_or_path \
        --index_dir $index_dir \
        --task_name indexing \
        --eval_batch_size 64 \
        --corpus_path $corpus_path \
        --alpha $alpha 
        #--out_ciff
if [[ $NGPU > 1 ]]; then 
    python -m scaling_retriever.utils.inverted_index \
        --model_name_or_path $model_name_or_path \
        --index_dir $index_dir
    else
        :
    fi