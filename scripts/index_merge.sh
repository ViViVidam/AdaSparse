#!/bin/bash
#SBATCH -A bgca-dtai-gh
#SBATCH --job-name="sparse indexing"
#SBATCH --output="./output/index-merge/top-k-per-scale-sparse-1B-%j.out"
#SBATCH --error="./output/index-smerge/top-k-per-scale-sparse-1B-%j.err"
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH -t 2:00:00



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

model_name_or_path=/projects/bfqn/yzound/checkpoint/lion/llama3-1b-adasparse-test

# eval_batch_size 64, because changed from A100 to A40
python -m scaling_retriever.utils.inverted_index \
	--model_name_or_path $model_name_or_path \
	--index_dir "/work/nvme/bcgk/yzound/index/msmarco/AdaSparse/index"