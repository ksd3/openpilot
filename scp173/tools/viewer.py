#!/usr/bin/env python3
"""SCP-173 local debug viewer.

Connects directly to the Comma 4's H.264 encoded camera stream via ZMQ
(same method as eye_hunter.py) — full framerate video without re-encoding.
State overlay arrives separately over UDP from main.py.

Usage:
  python scp173/tools/viewer.py --addr 192.168.63.120

Requirements (host machine venv):
  pip install pyzmq av opencv-python numpy
"""

import argparse
import json
import socket
import struct
import threading
import time

import av
import cv2
import numpy as np
import zmq

from cereal import log

# ── Cereal ZMQ port derivation (matches openpilot messaging) ─────────────────

def _cereal_port(endpoint: str) -> int:
    fnv_prime = 0x100000001b3
    h = 0xcbf29ce484222325
    for c in endpoint.encode():
        h ^= c
        h = (h * fnv_prime) & 0xFFFFFFFFFFFFFFFF
    return 8023 + (h % (65535 - 8023))


STREAMS = {
    "road":   "roadEncodeData",
    "wide":   "wideRoadEncodeData",
    "driver": "driverEncodeData",
}

# ── Shared state ──────────────────────────────────────────────────────────────

latest_frame: np.ndarray | None = None
frame_lock = threading.Lock()

latest_meta: dict = {}
meta_lock = threading.Lock()


# ── Video decode thread ───────────────────────────────────────────────────────

def video_thread(addr: str, stream_name: str) -> None:
    global latest_frame

    sock_name = STREAMS[stream_name]
    port = _cereal_port(sock_name)
    endpoint = f"tcp://{addr}:{port}"

    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.RCVHWM, 10)
    sock.connect(endpoint)

    print(f"[video] Connecting to {stream_name} stream at {endpoint}")

    codec = av.CodecContext.create("h264", "r")
    seen_iframe = False

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    while True:
        if not poller.poll(5000):
            print(f"[video] No frames for 5s — is bridge/encoderd running on the Comma 4?")
            continue

        data = sock.recv()

        try:
            with log.Event.from_bytes(data) as evt:
                enc = getattr(evt, evt.which())
                pkt_data = bytes(enc.data)
                is_iframe = bool(enc.idx.flags & 8)  # V4L2_BUF_FLAG_KEYFRAME
        except Exception:
            continue

        if not seen_iframe:
            if not is_iframe:
                continue
            seen_iframe = True

        try:
            packets = codec.parse(pkt_data)
            for pkt in packets:
                frames = codec.decode(pkt)
                for f in frames:
                    bgr = f.to_ndarray(format="bgr24")
                    with frame_lock:
                        latest_frame = bgr
        except Exception:
            seen_iframe = False  # resync on decode error


# ── Metadata receiver thread (UDP from main.py) ───────────────────────────────

def meta_thread(port: int) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(1.0)
    while True:
        try:
            data, _ = sock.recvfrom(65536)
            if len(data) < 8:
                continue
            meta_len = struct.unpack_from(">I", data, 4)[0]
            meta_bytes = data[8: 8 + meta_len]
            with meta_lock:
                latest_meta.update(json.loads(meta_bytes.decode()))
        except socket.timeout:
            pass
        except Exception:
            pass


# ── Main display loop ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr",   default="192.168.63.120", help="Comma 4 IP")
    parser.add_argument("--camera", default="road", choices=["road", "wide", "driver"])
    parser.add_argument("--port",   type=int, default=5174, help="UDP metadata port")
    args = parser.parse_args()

    # Start background threads
    threading.Thread(target=video_thread, args=(args.addr, args.camera), daemon=True).start()
    threading.Thread(target=meta_thread,  args=(args.port,),             daemon=True).start()

    cv2.namedWindow("SCP-173", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SCP-173", 1280, 720)

    fps_t   = time.monotonic()
    fps_cnt = 0
    disp_fps = 0.0

    COLOURS = {
        "IDLE":     (128, 128, 128),
        "STALKING": (0,   165, 255),
        "FROZEN":   (255,   0,   0),
        "STRIKE":   (0,     0, 255),
    }

    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            blank = np.zeros((480, 854, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for camera stream...",
                        (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 200), 2)
            cv2.imshow("SCP-173", blank)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
            continue

        with meta_lock:
            meta = dict(latest_meta)

        # FPS counter
        fps_cnt += 1
        now = time.monotonic()
        if now - fps_t >= 1.0:
            disp_fps = fps_cnt / (now - fps_t)
            fps_cnt = 0
            fps_t = now

        # ── Overlay ───────────────────────────────────────────────────
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)

        state  = meta.get("state", "---")
        colour = COLOURS.get(state, (255, 255, 255))
        cv2.putText(frame, f"[{state}]", (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)

        info = (
            f"road:{meta.get('road',0)} rear:{meta.get('rear',0)}  "
            f"src:{meta.get('src','?')}  "
            f"watched:{meta.get('watched','?')}  "
            f"dist:{meta.get('dist',0):.3f}  "
            f"cmd:({meta.get('cmd',[0,0])[0]:.2f},{meta.get('cmd',[0,0])[1]:+.2f})  "
            f"ml:{meta.get('fps',0):.1f}fps  view:{disp_fps:.1f}fps"
        )
        cv2.putText(frame, info, (155, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("SCP-173", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
