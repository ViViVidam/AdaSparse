"""
Analyze which tokens in the vocabulary appears in the index. Loading index_dist.json is fast.
Prints statistics about the df of those tokens.
"""

import json
import pickle
import numpy as np

fd = "/work/hdd/bcgk/zwang48/sclr_embedding/new/sparse-nolimit-1B-dfflops2-C30/beir/scifact/index_dist.json"
dist = json.load(open(fd, 'r'))
df = pickle.load(open("ms-marco-df.pkl",'rb'))
df_values = df.numpy()
print(dist["123"])

idxs = []
for token_idstr in dist.keys():
    token_id = int(token_idstr)
    idxs.append(token_id)
idxs = np.array(idxs)

df_appears = df_values[idxs]
print(f"Total tokens appears: {len(df_appears)}")
print(f"Max df: {df_appears.max()}")
print(f"Mean df: {df_appears.mean():.2f}")
print(f"Median df: {np.median(df_appears):.2f}")
print(f"25th percentile df: {np.percentile(df_appears, 25):.2f}")
print(f"50th percentile df: {np.percentile(df_appears, 50):.2f}")
print(f"55th percentile df: {np.percentile(df_appears, 55):.2f}")
print(f"90th percentile df: {np.percentile(df_appears, 90):.2f}")
print(f"99th percentile df: {np.percentile(df_appears, 99):.2f}")