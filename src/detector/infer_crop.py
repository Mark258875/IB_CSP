from ultralytics import YOLO
import cv2, json, os, argparse
from pathlib import Path

def save_crop(img, xyxy, out_dir, img_id, det_id):
    x1,y1,x2,y2 = map(int, xyxy)
    crop = img[y1:y2, x1:x2]
    out = Path(out_dir)/f"{img_id}_{det_id}.jpg"
    cv2.imwrite(str(out), crop)
    return str(out)

def main(weights, img_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    model = YOLO(weights)
    manifest = []
    for p in sorted(Path(img_dir).glob("*.jpg")):
        r = model(source=str(p), imgsz=1280, conf=0.25, iou=0.6, device=0, verbose=False)[0]
        img = cv2.imread(str(p))
        for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
            crop_path = save_crop(img, box, out_dir, p.stem, i)
            manifest.append({
              "image_path": str(p),
              "crop_path": crop_path,
              "bbox_xyxy": box.tolist(),
              "score": float(r.boxes.conf[i].cpu().item())
            })
    with open(Path(out_dir)/"crops.json","w") as f: json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])
