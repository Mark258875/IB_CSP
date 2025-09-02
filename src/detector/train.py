from ultralytics import YOLO
import yaml, argparse, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/detector.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg, "r"))

    model_path = cfg.get("model", "yolov8s.pt")
    data       = cfg.get("data",  "configs/data.yaml")
    imgsz      = int(cfg.get("imgsz", 1280))
    epochs     = int(cfg.get("epochs", 80))
    batch      = int(cfg.get("batch", 16))
    patience   = int(cfg.get("patience", 25))

    # device: 0 for first GPU, or 'cpu'
    device_cfg = cfg.get("device", "auto")
    device = 0 if (device_cfg == "auto" and torch.cuda.is_available()) else device_cfg
    if device == "auto":  # no GPU available
        device = "cpu"

    model = YOLO(model_path)
    model.train(
        data=data,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        device=device,
        cache=True,
        patience=patience,
        # augmentations (safe, helpful defaults)
        hsv_h=cfg.get("hsv_h", 0.02),
        hsv_s=cfg.get("hsv_s", 0.5),
        hsv_v=cfg.get("hsv_v", 0.5),
        degrees=cfg.get("degrees", 15),
        translate=cfg.get("translate", 0.1),
        scale=cfg.get("scale", 0.6),
        shear=cfg.get("shear", 4),
        perspective=cfg.get("perspective", 0.0005),
        mosaic=cfg.get("mosaic", 1.0),
        close_mosaic=cfg.get("close_mosaic", 10),
    )

if __name__ == "__main__":
    main()
