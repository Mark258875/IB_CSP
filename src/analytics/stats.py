import json, numpy as np, pandas as pd
from collections import defaultdict

# inputs: crops.json (with GPS per source image), labels.npy (cluster ids)
def compute(crops_json, labels_npy, gps_map_json, out_csv):
    crops = json.load(open(crops_json))
    labels = np.load(labels_npy)
    gps_map = json.load(open(gps_map_json))  # image_path -> {lat, lon}
    rows = []
    for i, c in enumerate(crops):
        img = c["image_path"]; cluster = int(labels[i])
        if cluster == -1: continue  # noise
        g = gps_map.get(img, None)
        if not g: continue
        rows.append({"cluster": cluster, "img": img, "lat": g["lat"], "lon": g["lon"]})
    df = pd.DataFrame(rows)
    if df.empty: return
    df["geobin"] = df.apply(lambda r: f'{round(r.lat,5)},{round(r.lon,5)}', axis=1)  # or geohash
    summary = (df.groupby("cluster")
                 .agg(sightings=("img","nunique"),
                      unique_locations=("geobin","nunique"))
                 .reset_index()
                 .sort_values(["unique_locations","sightings"], ascending=False))
    summary.to_csv(out_csv, index=False)
