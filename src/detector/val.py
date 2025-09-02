from ultralytics import YOLO

# change if your checkpoint is elsewhere
MODEL = "runs/detect/train2/weights/best.pt"

if __name__ == "__main__":
    m = YOLO(MODEL)
    # evaluates on the 'test' split from configs/data.yaml
    metrics = m.val(data="configs/data.yaml", split="test", imgsz=1024, batch=8, device=0)
    # metrics has map50, map50_95, precision, recall, etc.
    print({k: float(v) for k, v in metrics.results_dict.items()})
