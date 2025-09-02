# src/detector/check_dataset.py
import yaml
from pathlib import Path

cfg = yaml.safe_load(open("configs/data.yaml", "r"))
root = Path(cfg["path"])

def count_split(split_key):
    sub = cfg.get(split_key)
    if not sub:
        return
    img_dir = root / sub                    # e.g., data/grafiti-dataset/train/images
    if not img_dir.exists():
        print(f"[{split_key}] MISSING: {img_dir}")
        return
    # labels dir sits next to images: data/grafiti-dataset/train/labels
    lbl_dir = img_dir.parent / "labels"

    imgs = [*img_dir.glob("*.jpg"), *img_dir.glob("*.jpeg"), *img_dir.glob("*.png")]
    lbls = list(lbl_dir.glob("*.txt"))

    print(f"{split_key}: images={len(imgs)}, labels={len(lbls)} -> {img_dir}")
    print("  sample:", [p.name for p in imgs[:3]])
    if not lbl_dir.exists():
        print(f"  [warn] labels dir missing: {lbl_dir}")
    elif len(lbls) == 0:
        print(f"  [warn] no label .txt files found in: {lbl_dir}")

for k in ("train", "val", "test"):
    count_split(k)
