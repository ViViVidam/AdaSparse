#!/bin/bash
#SBATCH -A bcgk-delta-cpu
#SBATCH --job-name="counting df of MS MARCO"
#SBATCH --output="./compute-df-%j.out"
#SBATCH --error="./compute-df-%j.err"
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --no-requeue
#SBATCH --mem=64G
#SBATCH -t 2:00:00

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

echo "in sh: stdout"
python count-msmarco-df.py