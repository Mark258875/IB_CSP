import numpy as np, hdbscan, json
from pathlib import Path

def cluster(emb_npy, min_cluster_size=6, min_samples=2):
    X = np.load(emb_npy).astype('float32')
    c = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        metric='euclidean',
                        cluster_selection_epsilon=0.05)
    labels = c.fit_predict(X)
    Path("outputs").mkdir(exist_ok=True)
    np.save("outputs/labels.npy", labels)
    return labels

if __name__ == "__main__":
    import sys; cluster(sys.argv[1])
