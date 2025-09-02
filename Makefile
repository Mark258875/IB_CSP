train-detector:
\tpython src/detector/train.py

crops:
\tpython src/detector/infer_crop.py models/best.pt data/images outputs/crops

embed:
\tpython src/embedder/infer_embed.py outputs/crops outputs/embeddings.npy

index:
\tpython src/index/build_faiss.py outputs/embeddings.npy models

cluster:
\tpython src/cluster/hdbscan_cluster.py outputs/embeddings.npy

stats:
\tpython src/analytics/stats.py outputs/crops/crops.json outputs/labels.npy data/gps.json outputs/summary.csv
