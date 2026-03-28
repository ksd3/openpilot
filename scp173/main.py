#!/usr/bin/env python3
"""SCP-173 Comma Body — main control loop."""

import json
import logging
import socket
import struct
import threading
import time
import numpy as np
import cv2

# Log to file so we can review after a run
logging.basicConfig(
    filename="/data/openpilot/scp173/scp173.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scp173")

from msgq.visionipc import VisionIpcClient, VisionStreamType

from openpilot.common.realtime import Ratekeeper
from openpilot.common.params import Params

from scp173.config import (
    CONTROL_HZ, STREAM_HOST, STREAM_PORT,
)
from scp173.perception.person_detector_mobilenet import PersonDetector
from scp173.perception.attention_detector_fast import AttentionDetector
from scp173.behavior.state_machine        import SCP173StateMachine, State
from scp173.control.motor_controller      import MotorController


# ── Camera helpers ────────────────────────────────────────────────────────────

def yuv_to_bgr(buf) -> np.ndarray:
    h, w, stride = buf.height, buf.width, buf.stride
    nv12_size = h * 3 // 2 * stride
    raw = np.frombuffer(buf.data, dtype=np.uint8, count=nv12_size)
    nv12 = raw.reshape((h * 3 // 2, stride))
    return cv2.cvtColor(np.ascontiguousarray(nv12[:, :w]), cv2.COLOR_YUV2BGR_NV12)


def connect_camera(stream_type: VisionStreamType) -> VisionIpcClient:
    client = VisionIpcClient("camerad", stream_type, True)
    while not client.connect(False):
        time.sleep(0.1)
    return client


def best_detection(people: list, frame_w: int, frame_h: int) -> tuple | None:
    if not people:
        return None
    best = max(people, key=lambda p: (p[2] - p[0]) * (p[3] - p[1]))
    x1, y1, x2, y2, _ = best
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return (cx / frame_w) * 2.0 - 1.0, cx, cy


# ── Fast command publisher ────────────────────────────────────────────────────

class CommandPublisher:
    """Publishes the latest motor command at 20 Hz so joystickd stays alive."""

    PUBLISH_HZ = 20

    def __init__(self, motor: MotorController):
        self._motor   = motor
        self._accel   = 0.0
        self._steer   = 0.0
        self._lock    = threading.Lock()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update(self, accel: float, steer: float) -> None:
        with self._lock:
            self._accel = accel
            self._steer = steer

    def stop(self) -> None:
        with self._lock:
            self._accel = 0.0
            self._steer = 0.0
        self._running = False

    def _loop(self) -> None:
        rk = Ratekeeper(self.PUBLISH_HZ, print_delay_threshold=None)
        while self._running:
            with self._lock:
                a, s = self._accel, self._steer
            self._motor.send(a, s)
            rk.keep_time()


# ── Debug frame streamer ──────────────────────────────────────────────────────

class FrameStreamer:
    """Sends JPEG frames + metadata to the viewer on the host machine via UDP."""

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._enabled = bool(host)

    def send(self, frame: np.ndarray, meta: dict) -> None:
        if not self._enabled:
            return
        try:
            small = cv2.resize(frame, (640, 480))
            _, jpeg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
            meta_bytes = json.dumps(meta).encode()
            jpeg_bytes = jpeg.tobytes()
            header = struct.pack(">II", 8 + len(meta_bytes) + len(jpeg_bytes), len(meta_bytes))
            packet = header + meta_bytes + jpeg_bytes
            if len(packet) <= 65507:
                self._sock.sendto(packet, (self._host, self._port))
        except Exception:
            pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    params = Params()
    params.put_bool("JoystickDebugMode", True)

    print("SCP-173 online. Waiting for joystickd to start...")
    for _ in range(30):
        if params.get_bool("JoystickDebugMode"):
            break
        params.put_bool("JoystickDebugMode", True)
        time.sleep(1)

    # Keep re-setting it in case manager clears it
    def keep_joystick_mode():
        while True:
            params.put_bool("JoystickDebugMode", True)
            time.sleep(2)
    threading.Thread(target=keep_joystick_mode, daemon=True).start()

    print("JoystickDebugMode set. Initialising subsystems...")

    detector  = PersonDetector(input_size=200)  # smaller = faster
    attention = AttentionDetector()
    motor     = MotorController()
    publisher = CommandPublisher(motor)
    streamer  = FrameStreamer(STREAM_HOST, STREAM_PORT)
    fsm       = SCP173StateMachine()

    print("Connecting to road camera...")
    road_cam = connect_camera(VisionStreamType.VISION_STREAM_ROAD)
    print("Camera connected. Beginning containment breach...")

    tick           = 0
    being_watched  = False
    smooth_bearing = 0.0
    smooth_accel   = 0.0
    BEARING_ALPHA  = 0.4   # 0 = ignore new, 1 = fully trust new reading
    ACCEL_ALPHA    = 0.3
    MAX_ACCEL      = 0.2
    MAX_STEER_CMD  = 0.3
    fps            = 0.0

    try:
        while True:
            loop_start = time.monotonic()

            # ── Frame ────────────────────────────────────────────────
            road_buf = road_cam.recv()
            if road_buf is None:
                publisher.update(0.0, 0.0)
                continue

            road_frame = yuv_to_bgr(road_buf)
            h, w = road_frame.shape[:2]

            # ── Detection ────────────────────────────────────────────
            road_people = detector.detect(road_frame)
            person_detected = len(road_people) > 0

            target_bearing  = 0.0
            target_distance = 1.0

            if person_detected:
                det = best_detection(road_people, w, h)
                if det:
                    bearing, cx, cy = det
                    target_bearing = bearing
                    best_person = max(road_people, key=lambda p: (p[2] - p[0]) * (p[3] - p[1]))
                    bbox_w = (best_person[2] - best_person[0]) / w
                    target_distance = max(0.01, 1.0 - bbox_w)

            # ── Attention ────────────────────────────────────────────
            being_watched, _ = attention.is_being_watched(road_frame)

            # ── State machine ────────────────────────────────────────
            raw_accel, raw_steer = fsm.update(
                person_detected, being_watched, target_bearing, target_distance
            )

            # ── Smoothing ────────────────────────────────────────────
            if raw_accel > 0:
                smooth_bearing = smooth_bearing * (1 - BEARING_ALPHA) + raw_steer * BEARING_ALPHA
                smooth_accel = smooth_accel * (1 - ACCEL_ALPHA) + raw_accel * ACCEL_ALPHA
            else:
                # Stopping — decay quickly
                smooth_bearing *= 0.5
                smooth_accel *= 0.3

            final_accel = min(smooth_accel, MAX_ACCEL)
            final_steer = max(-MAX_STEER_CMD, min(MAX_STEER_CMD, smooth_bearing))

            log.info(
                "state=%s road=%d watched=%s "
                "bearing=%.2f dist=%.3f accel=%.2f steer=%+.2f raw_steer=%+.2f",
                fsm.state.name, len(road_people),
                being_watched,
                target_bearing, target_distance, final_accel, final_steer, raw_steer,
            )

            publisher.update(final_accel, final_steer)

            # ── Timing ───────────────────────────────────────────────
            elapsed = time.monotonic() - loop_start
            fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 0.001))

            streamer.send(road_frame, {
                "state":   fsm.state.name,
                "watched": being_watched,
                "dist":    round(target_distance, 3),
                "cmd":     [round(final_accel, 2), round(final_steer, 2)],
                "road":    len(road_people),
                "fps":     round(fps, 1),
            })

            print(
                f"[{fsm.state.name:8s}] "
                f"road:{len(road_people)} watched={being_watched} "
                f"cmd=({final_accel:.2f},{final_steer:+.2f}) "
                f"dist={target_distance:.3f} {elapsed*1000:.0f}ms"
            )

            tick += 1

    finally:
        publisher.stop()


if __name__ == "__main__":
    main()
