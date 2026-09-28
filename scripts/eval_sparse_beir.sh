#!/bin/bash

# subsets like "arguana" requires bigger memory to avoid performance issues

# CHANGE THIS
model_name_or_path=/projects/bcgk/yzound/checkpoint/lion/llama3-1b-0.408/
index_path=/projects/bgca/yzound/index/beir/lion/lion_1B_0.65_alpha/
out_path=/work/hdd/bcgk/yzound/result/lion_1B_0.65_alpha/
echo $index_path
echo $model_name_or_path
# "arguana" "fiqa" "nfcorpus" "quora" "scidocs" "scifact" "trec-covid" "webis-touche2020" "climate-fever" "dbpedia-entity"  "fever"  "hotpotqa" 
for subset in "arguana" "fiqa" "nfcorpus" "quora" "scidocs" "scifact" "trec-covid" "webis-touche2020" "climate-fever" "dbpedia-entity"  "fever"  "hotpotqa" "nq"; do 
    
    echo ${subset}
    # CHANGE THIS
    # /projects/bcgk/zwang48/sclr/sparse-1B-200/beir/$subset
    # /work/hdd/bcgk/zwang48/sclr_embedding/new/sparse-nolimit-1B-clkd/beir/$subset
    out_dir=$out_path/$subset
    beir_dataset_dir=/work/hdd/bcgk/zwang48/beir_dataset/$subset

    # -- CHANGE THIS: --sparsity_limit 200 ?
    torchrun --nproc_per_node=1 --master_port 4420 -m eval_sparse \
            --model_name_or_path $model_name_or_path \
            --index_dir $index_path/$subset/index \
            --out_dir $out_dir \
            --is_beir \
            --beir_dataset $subset \
            --beir_dataset_dir $beir_dataset_dir\
            --task_name retrieval \
            --top_k 1000

    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Error performing retrieval"
    fi

    python -m eval_sparse \
            --task_name evaluate_beir \
            --beir_dataset $subset \
            --beir_dataset_dir $beir_dataset_dir\
            --out_dir $out_dir 

done
python analysis/beir_results.py --base_dir $out_path