#many labels in the dataset - collapse into one

# tools/collapse_to_single_class.py
from pathlib import Path

root = Path("data/grafiti-dataset")
splits = ["train", "valid", "test"]

changed = 0
for s in splits:
    lbl_dir = root / s / "labels"
    if not lbl_dir.exists(): 
        print(f"[skip] {lbl_dir}"); 
        continue
    for p in lbl_dir.glob("*.txt"):
        lines_out = []
        txt = p.read_text(encoding="utf-8", errors="ignore")
        for L in txt.splitlines():
            L = L.strip()
            if not L:
                continue
            parts = L.split()
            parts[0] = "0"  # force class id to 0
            lines_out.append(" ".join(parts))
        p.write_text("\n".join(lines_out), encoding="utf-8")
        changed += 1

print(f"Rewrote {changed} label files to class 0.")
