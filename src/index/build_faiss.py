# src/index/build_faiss.py
import sys, json
from pathlib import Path
import numpy as np, faiss, math

def build(emb_npy, out_dir, kind="auto"):
    x = np.load(emb_npy).astype("float32")
    assert x.ndim == 2, f"embeddings shape must be (N,d), got {x.shape}"
    N, d = x.shape

    # If you L2-normalized embeddings, cosine ~= inner product; use IP for best retrieval.
    use_ip = True
    if use_ip:
        # ensure normalized (defensive)
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        x = x / norms

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if kind == "flatip" or (kind == "auto" and N < 10000):
        # Small/medium datasets: Flat or HNSW
        # For really tiny N, FlatIP is perfect.
        index = faiss.IndexFlatIP(d) if use_ip else faiss.IndexFlatL2(d)
        index.add(x)
        out = out_dir / "faiss_flat.index"
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, 32)  # M=32
        if use_ip:
            faiss.cvar.indexHNSWDefaultLevel = 1
            index.metric_type = faiss.METRIC_INNER_PRODUCT
        index.add(x)
        out = out_dir / "faiss_hnsw.index"
    else:
        # Large datasets: IVF-PQ with reasonable nlist
        nlist = max(1, int(4 * math.sqrt(N)))       # e.g., N=100k -> ~1264
        nlist = min(nlist, N)                       # must be <= N
        m = 64                                      # PQ codebooks; good for d=768
        quantizer = faiss.IndexFlatIP(d) if use_ip else faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)  # 8 bits/codebook
        index.train(x)                               # requires N ≥ nlist
        index.add(x)
        out = out_dir / "faiss_ivfpq.index"

    faiss.write_index(index, str(out))
    print(f"Built index: {out}  | N={N}, d={d}")

if __name__ == "__main__":
    emb = sys.argv[1]
    out = sys.argv[2]
    kind = sys.argv[3] if len(sys.argv) > 3 else "auto"
    build(emb, out, kind)
