#!/bin/bash

task_name=index_and_retrieval
corpus_path="path to collection.tsv"
# encoded query vectors are written to a per-dataset subdirectory of this base
out_base="path to query encoding output directory"

if [ $task_name = index_and_retrieval ]; then
    # CHANGE HERE
    model_name_or_path=hzeng/Lion-SP-8B-llama3-marco-mntp
    echo $model_name_or_path
    # Keep the file-name suffixes below: the branches in the loop match on them.
    query_paths=(
        "path to queries.small.dev.tsv"
        "path to msmarco-test2019-queries.tsv"
        "path to msmarco-test2020-queries.tsv"
    )

    for query_path in "${query_paths[@]}"; do
        if [[ $query_path == *all-train.csv ]]; then
            set_name=train
            ds_name=all
        elif [[ $query_path == *2019-queries.tsv ]]; then
            set_name=test
            ds_name=trec_dl_19
        elif [[ $query_path == *2020-queries.tsv ]]; then
            set_name=test
            ds_name=trec_dl_20
        elif [[ $query_path == *queries.*dev.tsv ]]; then
            set_name=dev
            ds_name=msmarco
        else
            echo "Error: Unknown set_name: $set_name"
            exit 1
        fi
        out_dir="$out_base/${ds_name}"

        # MAY NEED TO CHANGE PORT to avoid collision
        torchrun --nproc_per_node=1 --master_port 44458 -m eval_sparse \
            --model_name_or_path $model_name_or_path \
            --out_dir "$out_dir" \
            --query_path "$query_path" \
            --task_name encode_query \
            --eval_batch_size 64 \
            --top_k 1000 \
            --quant \
            --out_ciff

        retVal=$?
        if [ $retVal -ne 0 ]; then
            echo "Error performing query encoding"
            exit $retVal
        fi
    done
fi
