#!/bin/bash
#SBATCH -A bgca-dtai-gh
#SBATCH --job-name="eval term thresh scaled sparse"
#SBATCH --output="./output/eval-sparse/per-thresh-nolimit-%j.out"
#SBATCH --error="./output/eval-sparse/per-thresh-nolimit-%j.err"
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH -t 2:00:00

# module purge
# module load nvhpc

# load conda env variables
# loading the profile of user zwang48 to inject require functions
__conda_setup="$('/u/yzound/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/u/yzound/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/u/yzound/miniconda3/profile.d/conda.sh"
    else
        export PATH="/u/yzound/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate sl

task_name=index_and_retrieval
corpus_path=/projects/bcgk/zwang48/sclr/msmarco-full/collection.tsv

if [ $task_name = index_and_retrieval ]; then
    # CHANGE HERE
    model_name_or_path=Johonson/adasparse-1B
    echo $model_name_or_path
    query_paths=(
        /projects/bcgk/yzound/datasets/msmarco/queries.small.dev.tsv
        /projects/bcgk/zwang48/sclr/msmarco-full/DL_2019/msmarco-test2019-queries.tsv
        /projects/bcgk/zwang48/sclr/msmarco-full/DL_2020/msmarco-test2020-queries.tsv
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
            # CHANGE HERE
            out_dir=/work/hdd/bcgk/yzound/result/lion_8B/${ds_name}/top_k_${top_k}

            # CHANGE HERE --index_dir, MAY NEED TO CHANGE PORT to avoid collision
            torchrun --nproc_per_node=1 --master_port 44452 -m eval_sparse \
                --model_name_or_path $model_name_or_path \
                --index_dir /work/nvme/bcgk/yzound/index/msmarco/adasparse/index \
                --out_dir $out_dir \
                --query_path $query_path \
                --task_name retrieval \
                --top_k $top_k
                #--use_eos

            retVal=$?
            if [ $retVal -ne 0 ]; then
                echo "Error performing retrieval"
                exit $retVal
            fi

            mv $out_dir/run.json $out_dir/${top_k}_run.json
            if [[ $ds_name == msmarco && $set_name == dev ]]; then
                eval_qrel_path=/projects/bcgk/zwang48/sclr/msmarco-full/dev_queries/dev_qrel.json
                eval_metric='["mrr_10","recall"]'
                eval_run_path=$out_dir/${top_k}_run.json
                python -m eval_sparse \
                    --eval_run_path $eval_run_path \
                    --eval_qrel_path $eval_qrel_path \
                    --eval_metric $eval_metric \
                    --out_dir $out_dir \
                    --task_name evaluate_msmarco
            elif [[ $ds_name == trec_dl_19 && $set_name == test ]]; then
                eval_qrel_path=/projects/bcgk/zwang48/sclr/msmarco-full/DL_2019/qrel.json
                eval_metrics='["ndcg_cut"]'
                eval_run_path=$out_dir/${top_k}_run.json
                python -m eval_sparse \
                    --eval_run_path $eval_run_path \
                    --eval_qrel_path $eval_qrel_path \
                    --eval_metric $eval_metrics \
                    --out_dir $out_dir \
                    --task_name evaluate_msmarco

                eval_qrel_path=/projects/bcgk/zwang48/sclr/msmarco-full/DL_2019/qrel_binary.json
                eval_metrics='["mrr_10","recall"]'
                eval_run_path=$out_dir/${top_k}_run.json
                python -m eval_sparse \
                    --eval_run_path $eval_run_path \
                    --eval_qrel_path $eval_qrel_path \
                    --eval_metric $eval_metrics \
                    --out_dir ${out_dir}_binary \
                    --task_name evaluate_msmarco
            elif [[ $ds_name == trec_dl_20 && $set_name == test ]]; then
                eval_qrel_path=/projects/bcgk/zwang48/sclr/msmarco-full/DL_2020/qrel.json
                eval_metrics='["ndcg_cut"]'
                eval_run_path=$out_dir/${top_k}_run.json
                python -m eval_sparse \
                    --eval_run_path $eval_run_path \
                    --eval_qrel_path $eval_qrel_path \
                    --eval_metric $eval_metrics \
                    --out_dir $out_dir \
                    --task_name evaluate_msmarco

                eval_qrel_path=/projects/bcgk/zwang48/sclr/msmarco-full/DL_2020/qrel_binary.json
                eval_metrics='["mrr_10","recall"]'
                eval_run_path=$out_dir/${top_k}_run.json
                python -m eval_sparse \
                    --eval_run_path $eval_run_path \
                    --eval_qrel_path $eval_qrel_path \
                    --eval_metric $eval_metrics \
                    --out_dir ${out_dir}_binary \
                    --task_name evaluate_msmarco
            else
                echo "Error: Unknown dataset: $ds_name"
                exit 1
            fi

            if [[ $set_name == train ]]; then
                python preprocess/create_de_self_train_data.py \
                    --q_ppid_npids_path "/work/hzeng_umass_edu/ir-research/GR-for-RAG-data/data/dpr-all/all-ret/query_pospid_negpids.train.jsonl" \
                    --retrieval_path $out_dir/rankings.${set_name}.jsonl \
                    --example_path $out_dir/de.train.jsonl
            fi
        done
    done
fi
