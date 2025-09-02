from ultralytics import YOLO
import sys

MODEL = sys.argv[1] if len(sys.argv) > 1 else "runs/detect/train2/weights/best.pt"
SOURCE = sys.argv[2] if len(sys.argv) > 2 else "data/images_all_city"  # folder, image, or video

if __name__ == "__main__":
    m = YOLO(MODEL)
    m.predict(
        source=SOURCE,
        imgsz=1024,
        conf=0.35,      # raise to 0.45 to reduce false positives; lower to catch more
        iou=0.6,        # NMS threshold
        device=0,       # GPU
        save=True,      # saves annotated images
        save_txt=True,  # YOLO-format detections in runs dir
        save_conf=True  # include confidences in txt
    )
