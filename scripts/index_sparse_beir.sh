#!/bin/bash
#SBATCH -A bgca-dtai-gh
#SBATCH --job-name="index BEIR"
#SBATCH --output="./output/index-beir-sparse/relaxed-8B-d-%j.out"
#SBATCH --error="./output/index-beir-sparse/relaxed-8B-d-%j.err"
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --no-requeue
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH -t 4:00:00

# "fever" "climate-fever"  requires bigger memory (to avoid OOM) 
# "dbpedia-entity" "hotpotqa" "webis-touche2020" need smaller batch size (to avoid CUDA OOM)
# otherwise, 62G Memory, 128 eval_batch_size is good

module purge
module load default
module load cuda/12.6.1
module load gcc/11.4.0 
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

# CHANGE THIS
# CL+KD 200: model_name_or_path=/projects/bcgk/zwang48/sclr/checkpoints/llama3-1b-marco-mntp-sparse-nce-kldiv-negs-Tevatron-lora-1e-4_qreg_05_dreg_04_bs_10_epochs_1_nnegs_16
# whose index dir is /projects/bcgk/zwang48/sclr/sparse-1B-clkd-tevatron/beir/, need --sparsity_limit 200
# CL+KD unlimited: model_name_or_path=/work/hdd/bcgk/zwang48/model_sclr_ckpts/llama3-1b-marco-mntp-sparse-nolimit-nce-kldiv-negs-Tevatron-lora-1e-4_qreg_05_dreg_04_bs_10_epochs_1_nnegs_16
# whose index dir is /work/hdd/bcgk/zwang48/sclr_embedding/new/sparse-nolimit-1B-clkd/beir/
model_name_or_path=/projects/bcgk/yzound/checkpoint/lion/llama3-1b-0.408/
index_path=/projects/bgca/yzound/index/beir/lion/lion_1B_0.65_alpha/
alpha=0.65
echo $model_name_or_path
echo $index_path
#"arguana" "fiqa" "nfcorpus" "quora" "scidocs" "scifact" "trec-covid" "webis-touche2020" "climate-fever" "dbpedia-entity" "fever" "hotpotqa" "nq" ;
NGPU=1
# - "climate-fever" "fever" "climate-fever"
# index dir /work/hdd/bcgk/zwang48/sclr_embedding/beir/sparse-1B/$subset
for subset in "arguana" "fiqa" "nfcorpus" "quora" "scidocs" "scifact" "trec-covid" "webis-touche2020"; do 
    out_path=$index_path/$subset/index
    echo ${subset}
    torchrun --nproc_per_node=$NGPU --master_port 4430 -m eval_sparse \
        --model_name_or_path $model_name_or_path \
        --task_name indexing \
        --is_beir \
        --beir_dataset $subset \
        --beir_dataset_dir /work/hdd/bcgk/zwang48/beir_dataset/$subset \
        --index_dir $out_path\
        --eval_batch_size 32 \
        --query_max_length 512 \
	    --doc_max_length 512 \
        --alpha $alpha

    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Error indexing" $subset
        exit $retVal
    fi

    python -m scaling_retriever.utils.inverted_index \
	    --model_name_or_path $model_name_or_path \
	    --index_dir $out_path
done