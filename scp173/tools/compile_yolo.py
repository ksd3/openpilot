#!/usr/bin/env python3
"""Pre-compile YOLOv8n for QCOM GPU using tinygrad.

Run once on the comma 4 device to generate optimized GPU kernels:

  PYTHONPATH=/data/openpilot /usr/local/venv/bin/python scp173/tools/compile_yolo.py

Output: scp173/models/yolov8n_qcom.pkl

Based on openpilot's compile3.py approach.
"""

import sys
import os
import pickle
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'tinygrad_repo'))

from tinygrad import Tensor, TinyJit, Device, GlobalCounters
from tinygrad.helpers import getenv

# Use native tinygrad YOLOv8 (not ONNX) for best optimization
from examples.yolov8 import YOLOv8, get_variant_multiples, get_weights_location, preprocess
from tinygrad.nn.state import safe_load, load_state_dict

OUTPUT = os.environ.get("OUTPUT", os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov8n_qcom.pkl'))
VARIANT = os.environ.get("YOLO_VARIANT", "n")
IMGSZ = int(os.environ.get("IMGSZ", "640"))


def compile_yolo():
    print(f"Device: {Device.DEFAULT}")
    print(f"Compiling YOLOv8{VARIANT} at {IMGSZ}px...")

    # Build model
    depth, width, ratio = get_variant_multiples(VARIANT)
    yolo = YOLOv8(w=width, r=ratio, d=depth, num_classes=80)

    # Load weights
    print("Loading weights...")
    weights_path = get_weights_location(VARIANT)
    state_dict = safe_load(weights_path)
    load_state_dict(yolo, state_dict)
    print("Weights loaded")

    # Wrap in TinyJit for compilation
    @TinyJit
    def run(img):
        return yolo(img).realize()

    # Warmup runs to trigger JIT compilation
    print("JIT compiling (this takes a minute)...")
    for i in range(3):
        GlobalCounters.reset()
        dummy = Tensor.randn(1, 3, IMGSZ, IMGSZ).realize()
        t0 = time.monotonic()
        out = run(dummy)
        out.numpy()
        t1 = time.monotonic()
        print(f"  warmup {i}: {(t1-t0)*1000:.0f}ms, {GlobalCounters.global_ops/1e9:.2f} GOPS")

    # Benchmark compiled version
    print("Benchmarking compiled model...")
    times = []
    for i in range(10):
        inp = Tensor.randn(1, 3, IMGSZ, IMGSZ).realize()
        t0 = time.monotonic()
        out = run(inp)
        out.numpy()
        t1 = time.monotonic()
        times.append(t1 - t0)

    avg = np.mean(times) * 1000
    print(f"Average: {avg:.0f}ms ({1000/avg:.1f} fps)")

    # Save compiled model
    print(f"Saving to {OUTPUT}...")
    with open(OUTPUT, 'wb') as f:
        pickle.dump(run, f)
    print(f"Done! Size: {os.path.getsize(OUTPUT) / 1024 / 1024:.1f}MB")


if __name__ == "__main__":
    compile_yolo()
