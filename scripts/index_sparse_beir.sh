#!/bin/bash

# "fever" "climate-fever"  requires bigger memory (to avoid OOM)
# "dbpedia-entity" "hotpotqa" "webis-touche2020" need smaller batch size (to avoid CUDA OOM)
# otherwise, 62G Memory, 128 eval_batch_size is good

# CHANGE HERE
model_name_or_path=Johonson/adasparse-1B
# per-subset index directories are created under this root
index_path="path to BEIR index output directory"
# BEIR subsets are downloaded here automatically on first use; no need to prepare them
beir_dataset_root=./data/beir_dataset
alpha=0.65
echo $model_name_or_path
echo "$index_path"
#"arguana" "fiqa" "nfcorpus" "quora" "scidocs" "scifact" "trec-covid" "webis-touche2020" "climate-fever" "dbpedia-entity" "fever" "hotpotqa" "nq" ;
NGPU=1
for subset in "arguana" "fiqa" "nfcorpus" "quora" "scidocs" "scifact" "trec-covid" "webis-touche2020"; do
    out_path="$index_path/$subset/index"
    echo ${subset}
    torchrun --nproc_per_node=$NGPU --master_port 4430 -m eval_sparse \
        --model_name_or_path $model_name_or_path \
        --task_name indexing \
        --is_beir \
        --beir_dataset $subset \
        --beir_dataset_dir "$beir_dataset_root/$subset" \
        --index_dir "$out_path" \
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
	    --index_dir "$out_path"
done
