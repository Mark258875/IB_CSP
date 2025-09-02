from ultralytics import YOLO
import yaml, argparse

def main(cfg="configs/detector.yaml"):
    args = yaml.safe_load(open(cfg))
    model = YOLO(args.get("model", "yolov8s.pt"))
    model.train(
        data=args.get("data","configs/data.yaml"),
        imgsz=args.get("imgsz",1280),
        epochs=args.get("epochs",100),
        batch=args.get("batch",16),
        mosaic=1.0, degrees=15, perspective=0.0005,
        hsv_h=0.02, hsv_s=0.5, hsv_v=0.5,
        shear=4, translate=0.1, scale=0.6,
        cache=True, device=0
    )

if __name__ == "__main__":
    main()
