import pickle

f_in = open("/work/hdd/bcgk/zwang48/sclr_embedding/new/sparse-idfscale-d/msmarco-dev-q/query_vecs.pkl",'rb')
obj = pickle.load(f_in)
query_ids, query_vectors = obj["qids"], obj["query_vecs"]
# id of first query
print(f"id of first query is {query_ids[0]}")
# sparse representation of first query
token_ids, values = query_vectors[0]
print(f"sparse vector length of the first query is {len(token_ids)}")
print("positions of non-zero elements:")
print(token_ids) # values correspond to token id (|V| is 128256)
print("values of non-zero elements:")
print(values) # weights for each value