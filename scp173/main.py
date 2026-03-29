#!/usr/bin/env python3
"""SCP-173 Comma Body — main control loop.

Uses BlazeFace (~24ms, 40fps) for combined person + attention detection.
Face visible = someone is watching = FREEZE.
Face disappears = they looked away = MOVE toward last known position.
"""

import logging
import subprocess
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

from cereal import messaging as cereal_messaging
from scp173.perception.detector_yunet import YuNetDetector
from scp173.behavior.state_machine import SCP173StateMachine, State
from scp173.control.motor_controller import MotorController

YUNET_MODEL = "/data/openpilot/scp173/models/face_detection_yunet_2023mar.onnx"
SOUND_STARTUP = "/data/openpilot/scp173/sounds/bewareilive.wav"
SOUND_CHASE = "/data/openpilot/scp173/sounds/aaaaaaa.wav"
SOUND_RUN = "/data/openpilot/scp173/sounds/run_coward.wav"


class SoundPlayer:
    """Non-blocking audio player using aplay."""

    def __init__(self):
        self._proc = None

    def play(self, path: str):
        """Play a sound, stopping any currently playing sound."""
        self.stop()
        try:
            self._proc = subprocess.Popen(
                ["aplay", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
            self._proc = None

    @property
    def is_playing(self):
        return self._proc is not None and self._proc.poll() is None


def yuv_to_bgr_small(buf, target_w: int = 320, target_h: int = 240) -> np.ndarray:
    """Convert NV12 to BGR at reduced resolution. ~2ms vs ~100ms for full res."""
    h, w, stride = buf.height, buf.width, buf.stride
    raw = np.frombuffer(buf.data, dtype=np.uint8, count=h * 3 // 2 * stride)
    nv12 = raw.reshape((h * 3 // 2, stride))[:, :w]
    y_small = cv2.resize(nv12[:h, :], (target_w, target_h))
    uv_small = cv2.resize(nv12[h:, :], (target_w, target_h // 2))
    return cv2.cvtColor(np.vstack([y_small, uv_small]), cv2.COLOR_YUV2BGR_NV12)


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
    import sys, os
    mute = "--mute" in sys.argv or os.environ.get("MUTE") == "1"
    print(f"Sound: {'OFF' if mute else 'ON'}")
    params = Params()
    # Set JoystickDebugMode to stop controlsd, then kill joystickd
    # We publish carControl directly with enabled=True
    params.put_bool("JoystickDebugMode", True)
    time.sleep(2)  # let manager swap controlsd → joystickd
    subprocess.run(["pkill", "-f", "joystickd"], capture_output=True)
    subprocess.run(["pkill", "-f", "soundd"], capture_output=True)
    time.sleep(0.5)
    print("Killed joystickd + soundd — we own carControl and audio")

    # Start sending commands IMMEDIATELY to prevent selfdrived from disengaging
    motor = MotorController()
    publisher = CommandPublisher(motor)
    print("Command publisher started (100Hz keepalive)")

    sound = SoundPlayer() if not mute else None
    if sound:
        sound.play(SOUND_STARTUP)

    print("SCP-173 online. Loading YuNet...")
    detector = YuNetDetector(YUNET_MODEL, input_size=(320, 240), conf_threshold=0.3)
    fsm = SCP173StateMachine()

    print("Connecting to road camera...")
    road_cam = connect_camera(VisionStreamType.VISION_STREAM_ROAD)
    sm = cereal_messaging.SubMaster(['carState', 'livePose'])
    print("Camera connected. Beginning containment breach...")

    # Smoothing
    smooth_bearing = 0.0
    smooth_accel = 0.0
    BEARING_ALPHA = 0.3
    ACCEL_ALPHA = 0.3
    MAX_ACCEL = 0.2
    MAX_STEER_CMD = 0.15   # gentle corrections only
    STEER_RATE_LIMIT = 0.05  # very smooth steering changes

    # Last known bearing for when face disappears
    last_known_bearing = 0.0
    frames_since_face = 0

    fps = 0.0
    tick = 0
    prev_state = State.IDLE
    last_commanding_time = 0.0
    last_actually_moving_time = 0.0
    backup_until = 0.0
    turn_until = 0.0
    turn_direction = 1.0

    # World-space position tracking
    pos_x = 0.0   # meters forward from start
    pos_y = 0.0   # meters right from start
    heading = 0.0  # radians, 0 = initial forward
    last_pose_time = time.monotonic()

    try:
        while True:
            t0 = time.monotonic()

            t_recv = time.monotonic()
            buf = road_cam.recv()
            if buf is None:
                publisher.update(0.0, 0.0)
                continue
            t_got_frame = time.monotonic()

            frame_bgr = yuv_to_bgr_small(buf, 320, 240)
            t_cvt = time.monotonic()

            h, w = frame_bgr.shape[:2]

            # YuNet face detection (~31ms at 320x240)
            faces, being_watched, bearing = detector.detect(frame_bgr)
            t_detect = time.monotonic()

            if tick % 20 == 0:
                log.info("TIMING recv=%.0fms cvt=%.0fms detect=%.0fms frame=%dx%d",
                    (t_got_frame - t_recv) * 1000,
                    (t_cvt - t_got_frame) * 1000,
                    (t_detect - t_cvt) * 1000,
                    w, h)

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
                target_distance = max(0.01, 1.0 - face_w * 3)
            else:
                target_distance = 0.8

            # Steer toward face when visible, briefly chase last known bearing, then go straight
            if bearing is not None:
                target_bearing = bearing
            elif frames_since_face < 5:
                target_bearing = last_known_bearing  # chase briefly after face disappears
            else:
                target_bearing = 0.0  # go straight after a few frames

            # State machine
            raw_accel, raw_steer = fsm.update(
                person_detected, being_watched, target_bearing, target_distance
            )

            # State transition handling
            cur_state = fsm.state
            if cur_state == State.STALKING and prev_state == State.FROZEN:
                # Reset smoothing so robot heads toward last known bearing cleanly
                smooth_bearing = last_known_bearing * 0.8  # start aimed at person
                smooth_accel = 0.0

            if sound:
                if cur_state != prev_state:
                    if cur_state == State.STALKING and prev_state == State.FROZEN:
                        sound.play(SOUND_RUN)
                    elif cur_state == State.STALKING and prev_state == State.IDLE:
                        sound.play(SOUND_CHASE)
                elif cur_state == State.STALKING and not sound.is_playing:
                    sound.play(SOUND_CHASE)
                elif cur_state != State.STALKING:
                    sound.stop()
            prev_state = cur_state

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

            # IDLE: slowly rotate to scan for people
            if cur_state == State.IDLE and not being_watched:
                final_accel = 0.0
                final_steer = 0.1

            # Stuck detection: if commanding forward but not moving, back up
            sm.update(0)
            speed = abs(sm['carState'].vEgo)
            now = time.monotonic()

            # Update world-space position from livePose
            sm.update(0)
            pose = sm['livePose']
            dt = now - last_pose_time
            last_pose_time = now

            # velocityDevice: x=forward, y=right, z=down (device frame)
            vx = pose.velocityDevice.x
            vy = pose.velocityDevice.y
            world_speed = np.sqrt(vx**2 + vy**2)

            # Angular velocity for heading
            yaw_rate = pose.angularVelocityDevice.z
            heading += yaw_rate * dt

            # Integrate position in world frame
            pos_x += (vx * np.cos(heading) - vy * np.sin(heading)) * dt
            pos_y += (vx * np.sin(heading) + vy * np.cos(heading)) * dt

            # Track commanding vs actual movement
            if final_accel > 0.05:
                last_commanding_time = now
            if world_speed > 0.05:
                last_actually_moving_time = now

            recently_commanding = (now - last_commanding_time) < 2.0
            not_actually_moving = (now - last_actually_moving_time) > 1.5

            if now < backup_until:
                # Phase 1: back up
                final_accel = -0.3
                final_steer = 0.0
            elif now < turn_until:
                # Phase 2: turn away from the wall
                final_accel = 0.0
                final_steer = 0.15 * turn_direction
                # If we spot a face during turn, stop turning
                if bearing is not None:
                    turn_until = 0.0
            elif recently_commanding and not_actually_moving and now > (backup_until + 1.0):
                # Stuck! Start backup sequence
                # Turn opposite to where we were steering
                turn_direction = -1.0 if smooth_bearing > 0 else 1.0
                backup_until = now + 0.8
                turn_until = now + 2.5  # back up 0.8s then turn ~1.7s
                log.info("STUCK — backup + turn (dir=%.0f)", turn_direction)

            publisher.update(final_accel, final_steer)

            elapsed = time.monotonic() - t0
            fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 0.001))

            log.info(
                "state=%s faces=%d watched=%s bearing=%.2f dist=%.3f "
                "accel=%.2f steer=%+.2f pos=(%.2f,%.2f) spd=%.2f fps=%.0f %dms",
                fsm.state.name, len(faces), being_watched,
                target_bearing, target_distance,
                final_accel, final_steer, pos_x, pos_y, world_speed,
                fps, elapsed * 1000,
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
