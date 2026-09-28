#!/bin/bash

task_name=index_and_retrieval
corpus_path="path to collection.tsv"
# index built by scripts/index_sparse.sh
index_dir="path to index directory"
# retrieval runs and metrics are written to a per-dataset subdirectory of this base
out_base="path to evaluation output directory"

# qrels used to score each run
dev_qrel_path="path to dev_qrel.json"
dl19_qrel_path="path to DL_2019/qrel.json"
dl19_qrel_binary_path="path to DL_2019/qrel_binary.json"
dl20_qrel_path="path to DL_2020/qrel.json"
dl20_qrel_binary_path="path to DL_2020/qrel_binary.json"

if [ $task_name = index_and_retrieval ]; then
    # CHANGE HERE
    model_name_or_path=Johonson/adasparse-1B
    echo $model_name_or_path
    # Keep the file-name suffixes below: the branches in the loop match on them.
    query_paths=(
        "path to queries.cleaned.dev.tsv"
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
        elif [[ $query_path == *.dev.tsv ]]; then
            set_name=dev
            ds_name=msmarco
        else
            echo "Error: Unknown set_name: $set_name"
            exit 1
        fi
        for top_k in 1000; do
            out_dir="$out_base/${ds_name}/top_k_${top_k}"

            # MAY NEED TO CHANGE PORT to avoid collision
            torchrun --nproc_per_node=1 --master_port 44452 -m eval_sparse \
                --model_name_or_path $model_name_or_path \
                --index_dir "$index_dir" \
                --out_dir "$out_dir" \
                --query_path "$query_path" \
                --task_name retrieval \
                --top_k $top_k
                #--use_eos

            retVal=$?
            if [ $retVal -ne 0 ]; then
                echo "Error performing retrieval"
                exit $retVal
            fi

            mv "$out_dir/run.json" "$out_dir/${top_k}_run.json"
            if [[ $ds_name == msmarco && $set_name == dev ]]; then
                eval_qrel_path="$dev_qrel_path"
                eval_metric='["mrr_10","recall"]'
                eval_run_path="$out_dir/${top_k}_run.json"
                python -m eval_sparse \
                    --eval_run_path "$eval_run_path" \
                    --eval_qrel_path "$eval_qrel_path" \
                    --eval_metric $eval_metric \
                    --out_dir "$out_dir" \
                    --task_name evaluate_msmarco
            elif [[ $ds_name == trec_dl_19 && $set_name == test ]]; then
                eval_qrel_path="$dl19_qrel_path"
                eval_metrics='["ndcg_cut"]'
                eval_run_path="$out_dir/${top_k}_run.json"
                python -m eval_sparse \
                    --eval_run_path "$eval_run_path" \
                    --eval_qrel_path "$eval_qrel_path" \
                    --eval_metric $eval_metrics \
                    --out_dir "$out_dir" \
                    --task_name evaluate_msmarco

                eval_qrel_path="$dl19_qrel_binary_path"
                eval_metrics='["mrr_10","recall"]'
                eval_run_path="$out_dir/${top_k}_run.json"
                python -m eval_sparse \
                    --eval_run_path "$eval_run_path" \
                    --eval_qrel_path "$eval_qrel_path" \
                    --eval_metric $eval_metrics \
                    --out_dir "${out_dir}_binary" \
                    --task_name evaluate_msmarco
            elif [[ $ds_name == trec_dl_20 && $set_name == test ]]; then
                eval_qrel_path="$dl20_qrel_path"
                eval_metrics='["ndcg_cut"]'
                eval_run_path="$out_dir/${top_k}_run.json"
                python -m eval_sparse \
                    --eval_run_path "$eval_run_path" \
                    --eval_qrel_path "$eval_qrel_path" \
                    --eval_metric $eval_metrics \
                    --out_dir "$out_dir" \
                    --task_name evaluate_msmarco

                eval_qrel_path="$dl20_qrel_binary_path"
                eval_metrics='["mrr_10","recall"]'
                eval_run_path="$out_dir/${top_k}_run.json"
                python -m eval_sparse \
                    --eval_run_path "$eval_run_path" \
                    --eval_qrel_path "$eval_qrel_path" \
                    --eval_metric $eval_metrics \
                    --out_dir "${out_dir}_binary" \
                    --task_name evaluate_msmarco
            else
                echo "Error: Unknown dataset: $ds_name"
                exit 1
            fi
        done
    done
fi
