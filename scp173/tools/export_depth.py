#!/usr/bin/env python3
"""Export Depth Anything V2 Small to ONNX.

Run this once on a machine that has the Depth-Anything-V2 repo cloned,
then copy depth_anything_v2_vits.onnx to scp173/models/ on the Comma 4.

Usage:
  # Clone the repo first:
  #   git clone https://github.com/DepthAnything/Depth-Anything-V2
  #
  # Download the vits weights from Hugging Face into the repo's checkpoints/ dir:
  #   mkdir -p Depth-Anything-V2/checkpoints
  #   wget -O Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth \
  #     https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth
  #
  # Then export:
  #   python scp173/tools/export_depth.py --repo ./Depth-Anything-V2
  #   python scp173/tools/export_depth.py --repo ./Depth-Anything-V2 --size 224
"""

import argparse
import os
import sys


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Path to Depth-Anything-V2 clone")
    parser.add_argument("--size", type=int, default=308, help="Input size (308 or 224)")
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__), "..", "models", "depth_anything_v2_vits.onnx"
        ),
        help="Output ONNX path",
    )
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    if not os.path.isdir(repo):
        raise SystemExit(f"Repo not found: {repo}")

    weights = os.path.join(repo, "checkpoints", "depth_anything_v2_vits.pth")
    if not os.path.exists(weights):
        raise SystemExit(
            f"Weights not found: {weights}\n\n"
            "Download them first:\n"
            f"  mkdir -p {os.path.join(repo, 'checkpoints')}\n"
            f"  wget -O {weights} \\\n"
            "    https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
        )

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Add the repo to path so we can import depth_anything_v2
    sys.path.insert(0, repo)

    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    print(f"Loading model weights from {weights}...")
    model = DepthAnythingV2(**MODEL_CONFIGS["vits"])
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()

    dummy = torch.zeros(1, 3, args.size, args.size)

    print(f"Exporting to ONNX (input size {args.size}x{args.size})...")
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["image"],
        output_names=["depth"],
        opset_version=18,
        dynamic_axes=None,
    )

    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
