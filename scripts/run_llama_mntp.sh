#!/bin/bash
#SBATCH -A bgca-dtai-gh
#SBATCH --job-name="mntp on llama 3.2 7B"
#SBATCH --output="./output/mntp/mntp-7B-%j.out"
#SBATCH --error="./output/mntp/mntp-7B-%j.err"
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --no-requeue
#SBATCH --gpus=2
#SBATCH --mem=96G
#SBATCH -t 46:00:00

# module purge
# module load nvhpc
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
torchrun --nproc_per_node=$NGPU -m run_mntp train_configs/mntp/meta_llama2_7b_msmarco.json