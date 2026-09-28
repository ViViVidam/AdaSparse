#!/bin/bash
#SBATCH -A bcgk-delta-gpu
#SBATCH --job-name="eval term thresh only sparse"
#SBATCH --output="./output/encode-sparse/query-encode-1B-nolimit-%j.out"
#SBATCH --error="./output/encode-sparse/query-encode-1B-nolimit-%j.err"
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH -t 0:30:00

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
ms_marco_idf_path=/u/yzound/lion/preprocess/ms-marco-idf.pkl

if [ $task_name = index_and_retrieval ]; then
    # CHANGE HERE
    model_name_or_path=hzeng/Lion-SP-8B-llama3-marco-mntp
    #model_name_or_path=/projects/bfqn/yzound/checkpoint/lion/llama3-8b-ce-kldiv-learned-top-10-8k-dq-bm25-lora-1e-4_qreg_05_dreg_04_bs_14_epochs_1_nnegs_16_dual_perterm_relaxed/
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
        elif [[ $query_path == *queries.*dev.tsv ]]; then 
            set_name=dev
            ds_name=msmarco
        else 
            echo "Error: Unknown set_name: $set_name"
            exit 1
        fi
        # CHANGE HERE
        out_dir=/work/nvme/bcgk/yzound/index/msmarco/lion_8B/index/tmp/${ds_name}

        # CHANGE HERE --index_dir, MAY NEED TO CHANGE PORT to avoid collision
        torchrun --nproc_per_node=1 --master_port 44458 -m eval_sparse \
            --model_name_or_path $model_name_or_path \
            --out_dir $out_dir \
            --query_path $query_path \
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
