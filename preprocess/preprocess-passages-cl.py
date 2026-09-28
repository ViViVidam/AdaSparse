"""
Convert MS MARCO passage training set from Tevatron's train.jsonl 
(https://huggingface.co/datasets/Tevatron/msmarco-passage/tree/main)
to format accepted by train_dense and train_sparse
"""

import json
import argparse

def convert_train_jsonl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            ex = json.loads(line)
            query = ex["query"]
            pos_passages = ex["positive_passages"]
            neg_passages = ex["negative_passages"]

            # Use only the first positive passage's docid
            if not pos_passages:
                continue  # skip examples with no positive passage
            pos_pid = pos_passages[0]["docid"]

            # Collect all negative passage docids
            neg_pids = [neg["docid"] for neg in neg_passages if "docid" in neg]

            out = {
                "question": query,
                "pos_pid": pos_pid,
                "neg_pids": neg_pids
            }
            fout.write(json.dumps(out) + "\n")

    print(f"Conversion complete: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to original train.jsonl")
    parser.add_argument("--output", required=True, help="Path to save converted output jsonl")
    args = parser.parse_args()

    convert_train_jsonl(args.input, args.output)
