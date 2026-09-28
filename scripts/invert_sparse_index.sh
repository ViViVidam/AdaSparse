#!/bin/bash
#SBATCH -A bcgk-delta-cpu
#SBATCH --job-name="merging index"
#SBATCH --output="./output/merge-index-1B-%j.out"
#SBATCH --error="./output/merge-index-1B-%j.err"
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --no-requeue
#SBATCH --mem=64G
#SBATCH -t 12:00:00

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


model_name_or_path=/projects/bcgk/zwang48/sclr/checkpoints/llama3-1b-marco-mntp-sparse-nce-lora-1e-4_qreg_05_dreg_04_bs_10_epochs_1_nnegs_16
python -m scaling_retriever.utils.inverted_index --model_name_or_path $model_name_or_path --index_dir /work/hdd/bcgk/zwang48/sclr_embedding/sparse-1B --index_name sparse