"""
Convert TREC DL 2019, 2020 files from https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019.html to format
accepted by eval_dense.py and eval_sparse.py
"""

import json

def convert_qrel_txt_to_json(qrel_txt_path, output_json_path):
    qrel = {}
    with open(qrel_txt_path, 'r') as f:
        for line in f:
            qid, _, docid, rel = line.strip().split()
            if qid not in qrel:
                qrel[qid] = {}
            qrel[qid][docid] = int(rel)
    
    with open(output_json_path, 'w') as f:
        json.dump(qrel, f, indent=2)

def convert_qrel_to_binary(qrel_json_path, output_json_path):
    with open(qrel_json_path, 'r') as f:
        qrel = json.load(f)

    qrel_binary = {
        qid: {docid: int(rel > 1) for docid, rel in doc_rels.items()} # modify the qrels file to change all 1 judgments to 0.
        for qid, doc_rels in qrel.items()
    }

    with open(output_json_path, 'w') as f:
        json.dump(qrel_binary, f, indent=2)

base_path = "/projects/bcgk/zwang48/sclr/msmarco-full/DL_2019/"
convert_qrel_txt_to_json(base_path+"2019qrels-pass.txt", base_path+"qrel.json")
convert_qrel_to_binary(base_path+"qrel.json", base_path+"qrel_binary.json")

base_path = "/projects/bcgk/zwang48/sclr/msmarco-full/DL_2020/"
convert_qrel_txt_to_json(base_path+"2020qrels-pass.txt", base_path+"qrel.json")
convert_qrel_to_binary(base_path+"qrel.json", base_path+"qrel_binary.json")