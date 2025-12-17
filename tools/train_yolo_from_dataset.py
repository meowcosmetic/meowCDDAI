"""
Train/Fine-tune YOLOv8 từ dataset đã annotate (format Ultralytics).

Ví dụ:
  python tools/train_yolo_from_dataset.py --data dataset/dataset.yaml --weights yolov8n.pt --epochs 50 --imgsz 640

Output: Ultralytics sẽ tạo folder runs/detect/<name>
"""

from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path dataset.yaml")
    ap.add_argument("--weights", default="yolov8n.pt", help="Starting weights (vd: yolov8n.pt)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="custom-finetune", help="Run name")
    ap.add_argument("--device", default=None, help="cuda / cpu / 0 ... (optional)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("❌ Chưa có ultralytics. Cài: pip install ultralytics>=8.0.0")
        print(f"   Lỗi: {e}")
        return 1

    model = YOLO(args.weights)
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
    )
    if args.device:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)
    print("✅ Train xong. Check thư mục runs/detect/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


