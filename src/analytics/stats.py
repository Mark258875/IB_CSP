# src/analytics/stats.py
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

USAGE = """Usage:
  python src/analytics/stats.py CROPS_JSON LABELS_NPY [GPS_JSON] OUT_CSV

Examples:
  # no GPS
  python src/analytics/stats.py outputs/crops/crops.json outputs/labels.npy outputs/summary.csv
  # with GPS
  python src/analytics/stats.py outputs/crops/crops.json outputs/labels.npy data/gps.json outputs/summary.csv
"""

def main():
    if len(sys.argv) not in (4, 5):
        print(USAGE); sys.exit(1)

    if len(sys.argv) == 4:
        crops_json, labels_npy, out_csv = sys.argv[1], sys.argv[2], sys.argv[3]
        gps_json = None
    else:
        crops_json, labels_npy, gps_json, out_csv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    crops = json.loads(Path(crops_json).read_text(encoding="utf-8"))
    labels = np.load(labels_npy)
    assert len(crops) == len(labels), f"Length mismatch: {len(crops)} crops vs {len(labels)} labels"

    gps_map = None
    if gps_json and Path(gps_json).exists():
        gps_map = json.loads(Path(gps_json).read_text(encoding="utf-8"))

    rows = []
    for i, c in enumerate(crops):
        cid = int(labels[i])
        if cid == -1:  # noise cluster
            continue
        img = c.get("image_path") or c.get("source") or c.get("crop_path")
        rec = {"cluster": cid, "img": img}
        if gps_map and img in gps_map:
            g = gps_map[img]
            rec["lat"] = g.get("lat")
            rec["lon"] = g.get("lon")
        rows.append(rec)

    if not rows:
        print("No clustered detections (all noise or empty).")
        Path(out_csv).write_text("")
        return

    df = pd.DataFrame(rows)
    grp = df.groupby("cluster", as_index=False)

    out = grp.agg(
        sightings=("img", "nunique"),
        crops=("img", "size")
    )

    # If GPS present, also compute unique locations (by rounded lat/lon)
    if "lat" in df.columns and "lon" in df.columns:
        df["geobin"] = df.apply(lambda r: f"{round(r.lat,5)},{round(r.lon,5)}", axis=1)
        locs = df.groupby("cluster", as_index=False).agg(unique_locations=("geobin", "nunique"))
        out = out.merge(locs, on="cluster", how="left")

    out = out.sort_values(["sightings", "crops"], ascending=False)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(out)} clusters.")

if __name__ == "__main__":
    main()
