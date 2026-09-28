#!/bin/bash
#SBATCH -A bcgk-delta-cpu
#SBATCH --job-name="sparse index merge"
#SBATCH --output="./output/index-beir-sparse/index-merge-beir-1B-%j.out"
#SBATCH --error="./output/index-beir-sparse/index-merge-beir-1B-%j.err"
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --no-requeue
#SBATCH --mem=64G
#SBATCH -t 24:00:00


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
corpus_path=/projects/bcgk/zwang48/sclr/msmarco-full/collection.tsv
model_name_or_path=/projects/bfqn/yzound/checkpoint/lion/llama3-1b-ce-kldiv-learned-top-6-5k-dq-lora-1e-4_qreg_05_dreg_04_bs_28_epochs_1_nnegs_16_dual_perterm/

# eval_batch_size 64, because changed from A100 to A40
# "arguana" "fiqa" "nfcorpus" "quora" "scidocs" "scifact" "trec-covid" "webis-touche2020"
for subset in "webis-touche2020"; do 
    index_dir=/projects/bgca/yzound/index/beir/lion/lion_1B_6_5/$subset/index
    python -m scaling_retriever.utils.inverted_index \
	    --model_name_or_path $model_name_or_path \
	    --index_dir $index_dir
done