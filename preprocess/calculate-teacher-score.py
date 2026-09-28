"""
Convert MS MARCO passage training set from Tevatron's train.jsonl 
(https://huggingface.co/datasets/Tevatron/msmarco-passage/tree/main)
to format accepted by train_dense and train_sparse with knowledge distillation loss
by calculating its score with the teacher model MiniLM-L6-v2.

Using 1xA100, it use 3200 queries / min, or 53 query / s
Using cpu, it takes ~20s / query
"""

import json
import argparse
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2', device='cuda')
print("Loaded MiniLM L6 v2", flush=True)


def convert_train_jsonl(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        lc = 0
        for line in fin: # 400782
            lc += 1
            if lc % 10000 == 0:
                print(lc, flush=True)
            ex = json.loads(line)
            query = ex["query"]
            pos_passages = ex["positive_passages"]
            neg_passages = ex["negative_passages"]

            # Use only the first positive passage's docid
            if not pos_passages:
                continue  # skip examples with no positive passage
            pos_pid = pos_passages[0]["docid"]

            # Collect all negative passage docids
            neg_pids = [neg["docid"] for neg in neg_passages if ("docid" in neg) and ("text" in neg)]
            neg_txts = [neg["text"] for neg in neg_passages if ("docid" in neg) and ("text" in neg)]

            # Score with MiniLM
            pos_score = model.predict([(query, pos_passages[0]["text"])])[0].astype(float)
            neg_scores = model.predict([(query, neg_txt) for neg_txt in neg_txts])
            neg_scores = [x.astype(float) for x in neg_scores]

            out = {
                "question": query,
                "pos_pid": pos_pid,
                "neg_pids": neg_pids,
                "pos_score": pos_score,
                "neg_scores": neg_scores
            }
            fout.write(json.dumps(out) + "\n")

    print(f"Conversion complete: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/projects/bcgk/zwang48/sclr/msmarco-full/train.jsonl", help="Path to original train.jsonl")
    parser.add_argument("--output", default="/projects/bcgk/zwang48/sclr/msmarco-full/teacher/train-tevatron-negs-teacher.jsonl", help="Path to save converted output jsonl")
    args = parser.parse_args()

    convert_train_jsonl(args.input, args.output)
