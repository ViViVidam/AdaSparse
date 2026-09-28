"""
Combine MiniLM-L-6-v2 scores with MS MARCO train to obtain input file for CL + KD (KL Divergence Loss) training
used for scripts/msmarco/llama_1b_sparse_lora_train_cl-kd.sh
"""

import json
import argparse
import pickle

def convert_train_jsonl(input_path, queries_path, scores_path, output_path):
    q_lookup = dict()
    for line in open(queries_path, 'r'):
        line = line.split('\t')
        qid = int(line[0])
        q_lookup[qid] = line[1]
    print("finished loading queries")
    ce_scores = pickle.load(open(scores_path, 'rb'))
    print("finished loading cross encoder scores")
    f_in = open(input_path, 'r')
    f_out = open(output_path, 'w')
    pid_errors = 0
    score_errors = 0
    total_queries = 0
    for line in f_in:
        total_queries += 1
        ex = json.loads(line)
        qid_str = ex["qid"]
        qid = int(qid_str)
        query = q_lookup[qid]
        try:
            pos_pid = ex["pos"][0]
            neg_pids = ex["neg"]["bm25"]
        except:
            pid_errors += 1
            continue
        try:
            pos_score = ce_scores[qid][pos_pid]
            neg_scores = [ce_scores[qid][neg_pid] for neg_pid in neg_pids]
        except:
            score_errors += 1
            continue
        out = {
            "question": query,
            "pos_pid": str(pos_pid),
            "neg_pids": [str(neg_pid) for neg_pid in neg_pids],
            "pos_score": pos_score,
            "neg_scores": neg_scores
        }
        f_out.write(json.dumps(out) + "\n")

    print(f"total {total_queries} queries, {pid_errors} missing passage, {score_errors} missing score")
    print(f"Conversion complete: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    base_path = "/projects/bcgk/zwang48/sclr/msmarco-full/teacher/"
    parser.add_argument("--input", default=base_path+"msmarco-hard-negatives.jsonl", help="Path to hard negatives (jsonl)")
    parser.add_argument("--queries",default=base_path+"queries/queries.train.tsv", help="Path to queries file")
    parser.add_argument("--scores", default=base_path+"cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl", help="Path to MiniLM-L-6-v2 scores (jsonl)")
    parser.add_argument("--output", default=base_path+"train-with-teacher.jsonl", help="Path to save converted output jsonl")
    args = parser.parse_args()

    convert_train_jsonl(args.input, args.queries, args.scores, args.output)
