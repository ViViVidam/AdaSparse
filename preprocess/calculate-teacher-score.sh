#!/bin/bash
#SBATCH -A bcgk-delta-gpu
#SBATCH --job-name="calc MiniLM scores Tevatron"
#SBATCH --output="../output/preprocess/miniLM-%j.out"
#SBATCH --error="../output/preprocess/miniLM-%j.err"
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH --mem=62G
#SBATCH -t 4:00:00

module purge
module load nvhpc

# load conda env variables
# loading the profile of user zwang48 to inject require functions
__conda_setup="$('/u/zwang48/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/u/zwang48/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/u/zwang48/miniconda3/profile.d/conda.sh"
    else
        export PATH="/u/zwang48/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate sclr

python calculate-teacher-score.py