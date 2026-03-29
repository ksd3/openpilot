#!/usr/bin/env python3
"""SCP-173 Comma Body — main control loop.

Uses BlazeFace (~24ms, 40fps) for combined person + attention detection.
Face visible = someone is watching = FREEZE.
Face disappears = they looked away = MOVE toward last known position.
"""

import logging
import threading
import time
import numpy as np
import cv2

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

from scp173.perception.detector_yunet import YuNetDetector
from scp173.behavior.state_machine import SCP173StateMachine, State
from scp173.control.motor_controller import MotorController

YUNET_MODEL = "/data/openpilot/scp173/models/face_detection_yunet_2023mar.onnx"


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


class CommandPublisher:
    PUBLISH_HZ = 100

    def __init__(self, motor: MotorController):
        self._motor = motor
        self._accel = 0.0
        self._steer = 0.0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update(self, accel: float, steer: float):
        with self._lock:
            self._accel = accel
            self._steer = steer

    def stop(self):
        with self._lock:
            self._accel = 0.0
            self._steer = 0.0
        self._running = False

    def _loop(self):
        rk = Ratekeeper(self.PUBLISH_HZ, print_delay_threshold=None)
        while self._running:
            with self._lock:
                a, s = self._accel, self._steer
            self._motor.send(a, s)
            rk.keep_time()


def main():
    params = Params()
    # Set JoystickDebugMode to stop controlsd, then kill joystickd
    # We publish carControl directly with enabled=True
    params.put_bool("JoystickDebugMode", True)
    import subprocess, time as _t
    _t.sleep(2)  # let manager swap controlsd → joystickd
    subprocess.run(["pkill", "-f", "joystickd"], capture_output=True)
    _t.sleep(0.5)
    print("Killed joystickd — we own carControl now")

    # Start sending commands IMMEDIATELY to prevent selfdrived from disengaging
    motor = MotorController()
    publisher = CommandPublisher(motor)
    print("Command publisher started (100Hz keepalive)")

    print("SCP-173 online. Loading YuNet...")
    detector = YuNetDetector(YUNET_MODEL, input_size=(320, 240), conf_threshold=0.3)
    fsm = SCP173StateMachine()

    print("Connecting to road camera...")
    road_cam = connect_camera(VisionStreamType.VISION_STREAM_ROAD)
    print("Camera connected. Beginning containment breach...")

    # Smoothing
    smooth_bearing = 0.0
    smooth_accel = 0.0
    BEARING_ALPHA = 0.3
    ACCEL_ALPHA = 0.3
    MAX_ACCEL = 0.2
    MAX_STEER_CMD = 0.3
    STEER_RATE_LIMIT = 0.15

    # Last known bearing for when face disappears
    last_known_bearing = 0.0
    frames_since_face = 0

    fps = 0.0
    tick = 0

    try:
        while True:
            t0 = time.monotonic()

            buf = road_cam.recv()
            if buf is None:
                publisher.update(0.0, 0.0)
                continue

            frame_bgr = yuv_to_bgr(buf)
            h, w = frame_bgr.shape[:2]

            # YuNet face detection (~31ms at 320x240)
            faces, being_watched, bearing = detector.detect(frame_bgr)

            if bearing is not None:
                last_known_bearing = bearing
                frames_since_face = 0
            else:
                frames_since_face += 1

            # Person detected = face seen recently (within last 10 frames)
            person_detected = frames_since_face < 10

            # Distance proxy from face size
            if faces:
                largest = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
                face_w = (largest[2] - largest[0]) / w
                target_distance = max(0.01, 1.0 - face_w * 3)  # faces are smaller than bodies
            else:
                target_distance = 0.8

            # Use last known bearing when face disappears (they turned away)
            target_bearing = last_known_bearing if person_detected else 0.0

            # State machine
            raw_accel, raw_steer = fsm.update(
                person_detected, being_watched, target_bearing, target_distance
            )

            # Smoothing + rate limiting
            if raw_accel > 0:
                new_bearing = smooth_bearing * (1 - BEARING_ALPHA) + raw_steer * BEARING_ALPHA
                delta = max(-STEER_RATE_LIMIT, min(STEER_RATE_LIMIT, new_bearing - smooth_bearing))
                smooth_bearing += delta
                smooth_accel = smooth_accel * (1 - ACCEL_ALPHA) + raw_accel * ACCEL_ALPHA
            else:
                smooth_bearing *= 0.5
                smooth_accel *= 0.3

            final_accel = min(smooth_accel, MAX_ACCEL)
            final_steer = max(-MAX_STEER_CMD, min(MAX_STEER_CMD, smooth_bearing))

            publisher.update(final_accel, final_steer)

            elapsed = time.monotonic() - t0
            fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 0.001))

            log.info(
                "state=%s faces=%d watched=%s bearing=%.2f dist=%.3f "
                "accel=%.2f steer=%+.2f fps=%.0f %dms",
                fsm.state.name, len(faces), being_watched,
                target_bearing, target_distance,
                final_accel, final_steer, fps, elapsed * 1000,
            )

            if tick % 10 == 0:
                print(
                    f"[{fsm.state.name:8s}] faces:{len(faces)} watched={being_watched} "
                    f"cmd=({final_accel:.2f},{final_steer:+.2f}) "
                    f"{elapsed*1000:.0f}ms {fps:.0f}fps"
                )

            tick += 1

    finally:
        publisher.stop()


if __name__ == "__main__":
    main()
