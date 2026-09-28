"""
Convert MS MARCO Dev qrel file from https://microsoft.github.io/msmarco/Datasets.html 
to format accepted by eval_dense.py and eval_sparse.py
"""

import json

def convert_qrels_tsv_to_json(qrels_path, output_path):
    qrel = {}
    with open(qrels_path, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            qid, _, pid, rel = line.strip().split()
            if qid not in qrel:
                qrel[qid] = {}
            qrel[qid][pid] = int(rel)

    with open(output_path, 'w') as f:
        json.dump(qrel, f, indent=2)

# Example usage:
base_path = "/projects/bcgk/zwang48/sclr/msmarco-full/dev_queries/"
convert_qrels_tsv_to_json(base_path+"qrels.dev.tsv", base_path+"dev_qrel.json")
