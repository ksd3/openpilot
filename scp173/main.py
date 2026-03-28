#!/usr/bin/env python3
"""SCP-173 Comma Body — main control loop."""

import json
import socket
import struct
import threading
import time
import numpy as np
import cv2

from msgq.visionipc import VisionIpcClient, VisionStreamType

from openpilot.common.realtime import Ratekeeper
from openpilot.common.params import Params

from scp173.config import (
    YOLO_MODEL_PATH, DEPTH_MODEL_PATH,
    CONTROL_HZ, STREAM_HOST, STREAM_PORT,
)
from scp173.perception.person_detector    import PersonDetector
from scp173.perception.attention_detector import AttentionDetector
from scp173.perception.depth_estimator    import DepthEstimator
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
            # Resize to 640x480 to keep UDP packet small
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

    print("SCP-173 online. Initialising subsystems...")

    detector  = PersonDetector(YOLO_MODEL_PATH)
    attention = AttentionDetector()
    depth_est = DepthEstimator(DEPTH_MODEL_PATH)
    motor     = MotorController()
    publisher = CommandPublisher(motor)
    streamer  = FrameStreamer(STREAM_HOST, STREAM_PORT)
    fsm       = SCP173StateMachine()

    print("Connecting to cameras...")
    road_cam   = connect_camera(VisionStreamType.VISION_STREAM_ROAD)
    wide_cam   = connect_camera(VisionStreamType.VISION_STREAM_WIDE_ROAD)
    driver_cam = connect_camera(VisionStreamType.VISION_STREAM_DRIVER)
    print("All cameras connected. Beginning containment breach...")

    tick          = 0
    driver_people = []
    fps_t         = time.monotonic()
    fps           = 0.0

    try:
        while True:
            loop_start = time.monotonic()

            # ── Frames ───────────────────────────────────────────────
            road_buf   = road_cam.recv()
            wide_buf   = wide_cam.recv()
            driver_buf = driver_cam.recv()

            if road_buf is None or wide_buf is None or driver_buf is None:
                publisher.update(0.0, 0.0)
                continue

            road_frame   = yuv_to_bgr(road_buf)
            wide_frame   = yuv_to_bgr(wide_buf)
            driver_frame = yuv_to_bgr(driver_buf)

            # ── Detection ────────────────────────────────────────────
            road_people = detector.detect(road_frame)
            if tick % 5 == 0:
                driver_people = detector.detect(driver_frame)

            person_detected = (len(road_people) + len(driver_people)) > 0

            target_bearing  = 0.0
            target_distance = 1.0
            depth_map       = None
            source_cam      = "none"

            road_det   = best_detection(road_people,   road_frame.shape[1], road_frame.shape[0])
            driver_det = best_detection(driver_people, driver_frame.shape[1], driver_frame.shape[0])

            if road_det:
                bearing, cx, cy = road_det
                target_bearing = bearing
                depth_map = depth_est.estimate(wide_frame)
                dx = int(np.clip((cx / road_frame.shape[1]) * depth_map.shape[1], 0, depth_map.shape[1] - 1))
                dy = int(np.clip((cy / road_frame.shape[0]) * depth_map.shape[0], 0, depth_map.shape[0] - 1))
                target_distance = float(depth_map[dy, dx])
                source_cam = "road"

            elif driver_det:
                bearing, _, _ = driver_det
                target_bearing  = -bearing
                target_distance = 0.8
                source_cam = "driver"

            # ── Attention ─────────────────────────────────────────────
            being_watched, _ = attention.is_being_watched(road_frame)

            # ── State machine ─────────────────────────────────────────
            raw_accel, raw_steer = fsm.update(
                person_detected, being_watched, target_bearing, target_distance
            )

            # Obstacle avoidance disabled — robot moves at full computed speed
            final_accel = raw_accel
            final_steer = raw_steer

            publisher.update(final_accel, final_steer)

            # ── Stream frame to viewer ────────────────────────────────
            elapsed = time.monotonic() - loop_start
            fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 0.001))
            streamer.send(road_frame, {
                "state":   fsm.state.name,
                "watched": being_watched,
                "dist":    round(target_distance, 3),
                "src":     source_cam,
                "cmd":     [round(final_accel, 2), round(final_steer, 2)],
                "road":    len(road_people),
                "rear":    len(driver_people),
                "fps":     round(fps, 1),
            })

            # ── Debug ─────────────────────────────────────────────────
            print(
                f"[{fsm.state.name:8s}] "
                f"road:{len(road_people)} rear:{len(driver_people)} "
                f"src={source_cam} watched={being_watched} "
                f"cmd=({final_accel:.2f},{final_steer:+.2f}) "
                f"dist={target_distance:.3f} {elapsed*1000:.0f}ms"
            )

            tick += 1

    finally:
        publisher.stop()


if __name__ == "__main__":
    main()
