#!/bin/bash

extract_task_weights() {
    local task_weights=$1

    # Remove the brackets and extract the individual values
    weights=$(echo $task_weights | tr -d '[]' | awk -F, '{print $2, $3}')

    # Assign the extracted values to the new variables and remove leading '.'
    local query_reg=$(echo $weights | awk '{print $1}' | sed 's/^\.//')
    local doc_reg=$(echo $weights | awk '{print $2}' | sed 's/^\.//')

    # Return the values as a string
    echo "$query_reg $doc_reg"
}

# train 
corpus_path="path to collection.tsv"
train_path="path to training dataset"
model_name_or_path="path to bidirectional model"

echo $teacher_score_path

# lr, task_weights, epochs, bz, n_negs, effective_batch_size
# bz=8, n_negs=16, can fit in to 4 A100 with lora_r=16
list_of_tuples=(
    1e-4 '[1.,.05,.04,1]' 1 28 16 112
)
NGPU=4

# scaling factors applied to doc/query bow lengths when computing the bow mask top-k
bow_doc_scale=8
bow_query_scale=10
# lexical provenance bias (beta in the paper): exempts original-text terms from STop gating
lexical_preserve=2

# It is unique for MS MARCO, when effective_batch_size equals to 512. 
# MS MARCO has 532,751 examples, so 1050 steps is around 1 epoch

for (( i=0; i<${#list_of_tuples[@]}; i+=${#list_of_tuples[@]} )); do
    lr=${list_of_tuples[i]}
    task_weights=${list_of_tuples[i+1]}
    epochs=${list_of_tuples[i+2]}
    batch_size=${list_of_tuples[i+3]}
    n_negs=${list_of_tuples[i+4]}
    effective_batch_size=${list_of_tuples[i+5]}
    
    gradient_accumulation_steps=$(($effective_batch_size / $NGPU / $batch_size))
    if [[ $effective_batch_size == 64 ]]; then 
        steps_per_epoch=8400 
    elif [[ $effective_batch_size == 512 ]]; then
        steps_per_epoch=1050
    elif [[ $effective_batch_size == 128 ]]; then 
        steps_per_epoch=4200
    elif [[ $effective_batch_size == 112 ]]; then 
        steps_per_epoch=4750
    else 
        echo "effective_batch_size is not supported"
        exit 1
    fi
    max_steps=$((steps_per_epoch * $epochs))
    save_steps=$((max_steps / 5))
    # wandb loss graph shows that ~600 steps is enough for loss to stabalize
    # although loss continue to drop on the log scale to the end (4750 steps)
    # max_steps=600
    # save_steps=10

    echo gradient_accumulation_steps: $gradient_accumulation_steps
    echo max_steps: $max_steps
    echo batch_size: $batch_size

    read query_reg doc_reg <<< $(extract_task_weights "$task_weights")
    run_name=llama3-1b-adasparse
    output_dir=./checkpoints/$run_name

    # save more checkpoints using save_total_limit
    torchrun --nproc_per_node=$NGPU --master_port 4426 -m train_sparse \
            --max_steps=$max_steps \
            --run_name=$run_name \
            --learning_rate=$lr \
            --model_name_or_path=$model_name_or_path \
            --output_dir=$output_dir \
            --task_names='["rank","query_reg","doc_reg","threshold_loss"]' \
            --task_weights=$task_weights \
            --wandb_project_name=llm_as_retriever \
            --bf16 \
            --query_max_length=64 \
            --doc_max_length=128 \
            --per_device_train_batch_size=$batch_size \
            --gradient_accumulation_steps=$gradient_accumulation_steps \
            --corpus_path=$corpus_path \
            --train_path=$train_path \
            --loss_type=adasparse \
            --logging_steps 50 \
            --warmup_ratio 0.04 \
            --save_steps $save_steps \
            --save_total_limit=2 \
            --gradient_checkpointing \
            --fsdp "full_shard auto_wrap" \
            --train_config train_configs/llama_config.json \
            --fsdp_config train_configs/fsdp_llama_config.json  \
            --lora \
            --model_type llama \
            --seed 45 \
            --n_negs $n_negs \
            --use_bow \
            --bow_doc_scale $bow_doc_scale \
            --bow_query_scale $bow_query_scale \
            --lexical_preserve $lexical_preserve \
            --thresh
done