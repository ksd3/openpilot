#!/usr/bin/env python3
"""Export YOLOv8-nano to ONNX for deployment on the Comma 4.

Run this once on a machine with pip install ultralytics, then copy
the resulting yolov8n.onnx to scp173/models/ on the device.

Usage:
  python scp173/tools/export_yolo.py
  python scp173/tools/export_yolo.py --size 416 --out scp173/models/yolov8n.onnx
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=416, help="Input image size")
    parser.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n.onnx"),
        help="Output ONNX path",
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("Install ultralytics first:  pip install ultralytics")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Exporting YOLOv8-nano to ONNX (imgsz={args.size})...")
    model = YOLO("yolov8n.pt")
    model.export(format="onnx", imgsz=args.size, opset=12, dynamic=False)

    # ultralytics exports to the same directory as the .pt file
    import shutil
    exported = "yolov8n.onnx"
    shutil.move(exported, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
