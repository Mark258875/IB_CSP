# IB_CSP
IB School Collaborative Science Project - AntiGrafiti layer

Trening...

TESTING (recognition): 

QUANTITATIVE TEST python src/detector/val.py
QUALITATIVE TEST (own pictures) python src/detector/predict.py
CROP (own pictures) python src/detector/infer_crop.py  runs/detect/train2/weights/best.pt "C:/Users/Marek Ištok/Documents/Visual studio code/IB_CSP/data/images_all_city" outputs/crops

DUPLICATEs:

python src/embedder/infer_embed.py outputs/crops outputs/embeddings.npy
python src/index/build_faiss.py outputs/embeddings.npy models
python src/cluster/hdbscan_cluster.py outputs/embeddings.npy
# optional: if you have GPS per source image
python src/analytics/stats.py outputs/crops/crops.json outputs/labels.npy data/gps.json outputs/summary.csv
# withoout GPS:
python src/analytics/stats.py outputs/crops/crops.json outputs/labels.npy outputs/summary.csv