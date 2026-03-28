#!/usr/bin/env python3
"""Camera feed verification — shows live BGR frames from all three cameras.

Run with camerad active (openpilot running, or tools/webcam/camerad.py).

Usage:
  python scp173/tools/test_cameras.py
  python scp173/tools/test_cameras.py --no-display   # headless, print stats only
"""

import argparse
import time
import numpy as np
import cv2

from msgq.visionipc import VisionIpcClient, VisionStreamType


CAMERAS = [
    ("Road",   VisionStreamType.VISION_STREAM_ROAD),
    ("Wide",   VisionStreamType.VISION_STREAM_WIDE_ROAD),
    ("Driver", VisionStreamType.VISION_STREAM_DRIVER),
]


def recv_bgr(client: VisionIpcClient):
    buf = client.recv()
    if buf is None:
        return None
    h, w, stride = buf.height, buf.width, buf.stride
    size = h * 3 // 2 * stride
    raw = np.frombuffer(buf.data, dtype=np.uint8, count=size).reshape((h * 3 // 2, stride))
    nv12 = np.ascontiguousarray(raw[:, :w])
    return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-display", action="store_true", help="Headless mode")
    args = parser.parse_args()

    clients = {}
    for name, stream in CAMERAS:
        c = VisionIpcClient("camerad", stream, True)
        print(f"Connecting to {name} camera...", end=" ", flush=True)
        while not c.connect(False):
            time.sleep(0.1)
        print("OK")
        clients[name] = c

    print("Press Q to quit.")
    frame_counts = {n: 0 for n in clients}
    t0 = time.monotonic()

    while True:
        frames = {}
        for name, client in clients.items():
            bgr = recv_bgr(client)
            if bgr is not None:
                frame_counts[name] += 1
                frames[name] = bgr

        elapsed = time.monotonic() - t0
        if elapsed > 2.0 and not args.no_display:
            fps_str = "  ".join(
                f"{n}: {frame_counts[n]/elapsed:.1f} fps" for n in clients
            )
            print(f"\r{fps_str}", end="", flush=True)

        if args.no_display:
            continue

        # Show a tiled view
        tiles = []
        for name in clients:
            if name in frames:
                img = cv2.resize(frames[name], (320, 240))
                cv2.putText(img, name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                tiles.append(img)

        if tiles:
            combined = np.hstack(tiles)
            cv2.imshow("SCP-173 cameras", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    print()


if __name__ == "__main__":
    main()
