"""
Compare the negative mining strategies of different sources of the MS MARCO passage retrieval dataset
First source: Tevatron (https://huggingface.co/datasets/Tevatron/msmarco-passage/tree/main)
Second source: Sentence Piece (https://huggingface.co/datasets/Tevatron/msmarco-passage/tree/main)
"""


import json
negs1 = "/projects/bcgk/zwang48/sclr/msmarco-full/train.jsonl"
negs2 = "/projects/bcgk/zwang48/sclr/msmarco-full/teacher/msmarco-hard-negatives.jsonl"

qid_my = 1000094 # 571018

qids1 = set()
count1 = 0
negs1count = 0
with open(negs1, 'r', encoding='utf-8') as n1f:
    for line in n1f:
        ex = json.loads(line)
        qid = int(ex["query_id"])
        neg_passages = ex["negative_passages"]
        if qid==qid_my:
            neg_pids1 = [int(neg["docid"]) for neg in neg_passages if "docid" in neg]
        qids1.add(qid)
        count1 += 1
        negs1count += len(neg_passages)
neg_pids1.sort()

print("done 1")

qids2 = set()
count2 = 0
negs2count = 0
with open(negs2, 'r', encoding='utf-8') as n2f:
    for line in n2f:
        ex = json.loads(line)
        qid = int(ex["qid"])
        if qid == qid_my:
            neg_pids2 = ex["neg"]["bm25"]
        qids2.add(qid)
        count2 += 1
        negs2count += len(neg_passages)
neg_pids2.sort()

print(f"num passages in Tevatron: {count1}, avg negs {negs1count/count1}, #queries deduplicated: {len(qids1)}")
print(f"num passages in Hard Negatives: {count2}, avg negs {negs2count/count2} #queries deduplicated: {len(qids2)}")

print(neg_pids1)
print(neg_pids2)
assert(neg_pids1==neg_pids2)