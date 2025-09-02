import faiss, numpy as np, json
from pathlib import Path

def build(emb_npy, out_dir):
    x = np.load(emb_npy).astype('float32')  # N x d (L2-normalized)
    d = x.shape[1]
    # IVF-PQ for scale; for <200k vectors HNSW or Flat is fine
    quantizer = faiss.IndexHNSWFlat(d, 32)
    index = faiss.IndexIVFPQ(quantizer, d, 4096, 64, 8)  # tweak for data size
    index.train(x)
    index.add(x)
    faiss.write_index(index, str(Path(out_dir)/"faiss_ivfpq.index"))
    print("Index written.")

if __name__ == "__main__":
    import sys; build(sys.argv[1], sys.argv[2])
