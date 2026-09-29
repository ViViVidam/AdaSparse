#!/bin/bash

# subsets like "arguana" requires bigger memory to avoid performance issues

# CHANGE HERE
model_name_or_path=Johonson/adasparse-1B
# per-subset index root written by scripts/index_sparse_beir.sh
index_path="path to BEIR index directory"
# retrieval runs and metrics are written to a per-subset subdirectory of this base
out_path="path to BEIR evaluation output directory"
# BEIR subsets are downloaded here automatically on first use; no need to prepare them
beir_dataset_root=./data/beir_dataset
echo "$index_path"
echo $model_name_or_path
# "arguana" "fiqa" "nfcorpus" "quora" "scidocs" "scifact" "trec-covid" "webis-touche2020" "climate-fever" "dbpedia-entity"  "fever"  "hotpotqa"
for subset in "arguana" "fiqa" "nfcorpus" "quora" "scidocs" "scifact" "trec-covid" "webis-touche2020" "climate-fever" "dbpedia-entity"  "fever"  "hotpotqa" "nq"; do

    echo ${subset}
    out_dir="$out_path/$subset"
    beir_dataset_dir="$beir_dataset_root/$subset"

    # -- CHANGE THIS: --sparsity_limit 200 ?
    torchrun --nproc_per_node=1 --master_port 4420 -m eval_sparse \
            --model_name_or_path $model_name_or_path \
            --index_dir "$index_path/$subset/index" \
            --out_dir "$out_dir" \
            --is_beir \
            --beir_dataset $subset \
            --beir_dataset_dir "$beir_dataset_dir" \
            --task_name retrieval \
            --top_k 1000

    retVal=$?
    if [ $retVal -ne 0 ]; then
        echo "Error performing retrieval"
    fi

    python -m eval_sparse \
            --task_name evaluate_beir \
            --beir_dataset $subset \
            --beir_dataset_dir "$beir_dataset_dir" \
            --out_dir "$out_dir"

done
python analysis/beir_results.py --base_dir "$out_path"
