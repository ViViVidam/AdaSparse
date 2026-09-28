#!/bin/bash
#SBATCH -A bgca-dtai-gh
#SBATCH --job-name="sparse indexing"
#SBATCH --output="./output/index-sparse/top-k-per-scale-sparse-1B-%j.out"
#SBATCH --error="./output/index-sparse/top-k-per-scale-sparse-1B-%j.err"
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --no-requeue
#SBATCH --gpus=2
#SBATCH --mem=128G
#SBATCH -t 16:00:00


# Unlimited sparse vector requires 128G for MS MARCO, otherwise 96G is enough

#module purge
module load cuda

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