#!/usr/bin/env python3
"""SCP-173 Comma Body — main control loop.

Uses BlazeFace (~24ms, 40fps) for combined person + attention detection.
Face visible = someone is watching = FREEZE.
Face disappears = they looked away = MOVE toward last known position.
"""

import logging
import math
import os
import shutil
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
from scp173.navigation.occupancy_grid import OccupancyGrid
from scp173.navigation import astar
from scp173.navigation.path_follower import PathFollower

YUNET_MODEL = "/data/openpilot/scp173/models/face_detection_yunet_2023mar.onnx"
KILLS_DIR = "/data/openpilot/scp173/kills"
SOUND_STARTUP = "/data/openpilot/scp173/sounds/bewareilive.wav"
SOUND_CHASE = "/data/openpilot/scp173/sounds/aaaaaaa.wav"
SOUND_RUN = "/data/openpilot/scp173/sounds/run_coward.wav"
SOUND_STRIKE = "/data/openpilot/scp173/sounds/necksnap.wav"
SOUND_IDLE = "/data/openpilot/scp173/sounds/concretegrind.wav"

FACE_STALKING = "/data/openpilot/scp173/assets/face_stalking.jpg"
FACE_FROZEN = "/data/openpilot/scp173/assets/face_frozen.jpg"
FACE_STRIKE = "/data/openpilot/scp173/assets/face_strike.jpg"
BG_IMAGE_PATH = "/usr/comma/bg.jpg"
BG_IMAGE_BACKUP = "/usr/comma/bg_original.jpg"


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


def take_kill_photo(buf, kill_count: int) -> str | None:
    """Save full-resolution trophy photo on STRIKE. Returns path or None."""
    try:
        os.makedirs(KILLS_DIR, exist_ok=True)
        h, w, stride = buf.height, buf.width, buf.stride
        raw = np.frombuffer(buf.data, dtype=np.uint8, count=h * 3 // 2 * stride)
        nv12 = raw.reshape((h * 3 // 2, stride))[:, :w]
        frame = cv2.cvtColor(np.ascontiguousarray(nv12), cv2.COLOR_YUV2BGR_NV12)

        text = f"KILL #{kill_count}"
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 6)
        cv2.putText(frame, ts, (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, "SCP-173", (40, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)

        path = os.path.join(KILLS_DIR, f"kill_{kill_count:03d}_{int(time.time())}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        log.info("KILL PHOTO saved: %s", path)
        return path
    except Exception as e:
        log.info("KILL PHOTO failed: %s", e)
        return None


class ScreenFace:
    """Swap the comma 4 background image to show SCP-173 face states."""

    def __init__(self):
        self._current = None
        # Backup original background
        if os.path.exists(BG_IMAGE_PATH) and not os.path.exists(BG_IMAGE_BACKUP):
            try:
                import shutil
                shutil.copy2(BG_IMAGE_PATH, BG_IMAGE_BACKUP)
            except Exception:
                pass

    def set_face(self, face_path: str):
        if face_path == self._current:
            return
        try:
            import shutil
            shutil.copy2(face_path, BG_IMAGE_PATH)
            # Kill and restart magic.py to reload the image
            subprocess.run(["pkill", "-f", "magic.py"], capture_output=True)
            self._current = face_path
        except Exception:
            pass

    def flash(self, face_path: str, times: int = 3):
        """Flash between face and black screen."""
        for _ in range(times):
            self.set_brightness(255)
            time.sleep(0.15)
            self.set_brightness(0)
            time.sleep(0.15)
        self.set_brightness(255)

    def set_brightness(self, val: int):
        try:
            with open("/sys/class/backlight/panel0-backlight/brightness", "w") as f:
                f.write(str(val))
        except Exception:
            pass


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
    os.environ["SENTRY_DSN"] = ""  # disable sentry error reporting
    os.environ["PYTHONWARNINGS"] = "ignore"
    mute = "--mute" in sys.argv or os.environ.get("MUTE") == "1"
    print(f"Sound: {'OFF' if mute else 'ON'}")
    params = Params()
    # Set JoystickDebugMode to stop controlsd, then kill joystickd
    # We publish carControl directly with enabled=True
    params.put_bool("JoystickDebugMode", True)
    time.sleep(2)  # let manager swap controlsd → joystickd
    subprocess.run(["pkill", "-f", "joystickd"], capture_output=True)
    subprocess.run(["pkill", "-f", "soundd"], capture_output=True)
    subprocess.run(["pkill", "-f", "loggerd"], capture_output=True)
    subprocess.run(["pkill", "-f", "encoderd"], capture_output=True)
    subprocess.run(["pkill", "-f", "uploader"], capture_output=True)
    subprocess.run(["pkill", "-f", "proclogd"], capture_output=True)
    time.sleep(0.5)
    print("Killed joystickd + soundd + logging — we own carControl and audio")

    # Start sending commands IMMEDIATELY to prevent selfdrived from disengaging
    motor = MotorController()
    publisher = CommandPublisher(motor)
    print("Command publisher started (100Hz keepalive)")

    sound = SoundPlayer() if not mute else None
    screen = ScreenFace()
    screen.set_face(FACE_STALKING)
    if sound:
        sound.play(SOUND_STARTUP)

    # Navigation
    nav_grid = OccupancyGrid(size=200, resolution=0.1)
    nav_follower = PathFollower(lookahead_distance=0.5)
    nav_replan_interval = 5  # replan every N frames
    nav_frame_count = 0

    print("SCP-173 online. Loading YuNet...")
    detector = YuNetDetector(YUNET_MODEL, input_size=(320, 240), conf_threshold=0.3)
    fsm = SCP173StateMachine()

    print("Connecting to wide camera...")
    road_cam = connect_camera(VisionStreamType.VISION_STREAM_WIDE_ROAD)
    sm = cereal_messaging.SubMaster(['carState', 'livePose'])
    print("Camera connected. Beginning containment breach...")

    # Smoothing
    smooth_bearing = 0.0
    smooth_accel = 0.0
    BEARING_ALPHA = 0.4
    ACCEL_ALPHA = 0.4
    MAX_ACCEL = 0.3
    MAX_STEER_CMD = 0.25
    STEER_RATE_LIMIT = 0.1

    # Person world-space memory
    last_known_bearing = 0.0
    last_face_time = 0.0
    ever_seen_someone = False
    HUNT_PERSISTENCE = 30.0
    WIDE_CAM_HALF_FOV = math.radians(60)  # wide camera ~120° FOV
    FACE_REAL_WIDTH = 0.17  # meters
    FOCAL_SCALED = 567.0 * (320.0 / 1928.0)  # wide camera focal length at 320px
    person_world_x = 0.0
    person_world_y = 0.0
    has_person_position = False

    fps = 0.0
    tick = 0
    prev_state = State.IDLE
    kill_count = 0
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
    prev_pos_x = 0.0
    prev_pos_y = 0.0

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
                last_face_time = now
                ever_seen_someone = True

                # Compute person's world position from bearing + face size
                largest = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
                face_w_px = largest[2] - largest[0]
                if face_w_px > 1:
                    distance_m = max(0.3, min(5.0, (FACE_REAL_WIDTH * FOCAL_SCALED) / face_w_px))
                else:
                    distance_m = 2.0
                person_angle = heading + bearing * WIDE_CAM_HALF_FOV
                person_world_x = pos_x + distance_m * math.cos(person_angle)
                person_world_y = pos_y + distance_m * math.sin(person_angle)
                has_person_position = True

            # Person detected = face seen recently (within persistence window)
            time_since_face = now - last_face_time if last_face_time > 0 else 999
            person_detected = time_since_face < HUNT_PERSISTENCE

            # Distance — use world-space distance if tracking, otherwise face size proxy
            if has_person_position and person_detected:
                dx = person_world_x - pos_x
                dy = person_world_y - pos_y
                target_distance = max(0.01, min(1.0, math.sqrt(dx**2 + dy**2) / 3.0))
            elif faces:
                face_w = (largest[2] - largest[0]) / w
                target_distance = max(0.01, 1.0 - face_w * 3)
            else:
                target_distance = 0.8

            # Steering: face visible → track face, face gone → A* path to world position
            if bearing is not None:
                target_bearing = bearing
                nav_follower.set_path([])  # clear path when we can see the face
            elif has_person_position and person_detected:
                # Plan/follow path using A*
                nav_frame_count += 1
                if nav_frame_count % nav_replan_interval == 0 or not nav_follower.has_path:
                    path = astar.plan(nav_grid, (pos_x, pos_y), (person_world_x, person_world_y))
                    if path:
                        nav_follower.set_path(path)
                        log.info("A* path: %d waypoints, grid obstacles=%d explored=%d",
                                 len(path), nav_grid.get_obstacle_count(), nav_grid.get_explored_count())

                if nav_follower.has_path:
                    target_bearing = nav_follower.get_steer(pos_x, pos_y, heading, WIDE_CAM_HALF_FOV)
                else:
                    # Fallback: direct bearing if A* fails
                    dx = person_world_x - pos_x
                    dy = person_world_y - pos_y
                    target_angle = math.atan2(dy, dx)
                    angle_diff = target_angle - heading
                    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
                    target_bearing = max(-1.0, min(1.0, angle_diff / WIDE_CAM_HALF_FOV))
            else:
                target_bearing = 0.0

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

            # Sound + screen on state transitions
            if cur_state != prev_state:
                if cur_state == State.FROZEN:
                    screen.set_face(FACE_FROZEN)
                    if sound: sound.stop()
                elif cur_state == State.STALKING:
                    screen.set_face(FACE_STALKING)
                    if sound:
                        sound.play(SOUND_RUN if prev_state == State.FROZEN else SOUND_CHASE)
                elif cur_state == State.STRIKE:
                    kill_count += 1
                    photo_path = take_kill_photo(buf, kill_count)
                    if sound: sound.play(SOUND_STRIKE)
                    log.info("STRIKE #%d at pos=(%.2f,%.2f) person=(%.2f,%.2f)",
                             kill_count, pos_x, pos_y, person_world_x, person_world_y)
                    # Show kill photo on screen (runs in background)
                    if photo_path:
                        subprocess.Popen([
                            "/usr/local/venv/bin/python",
                            "/data/openpilot/scp173/control/show_photo.py",
                            photo_path, "3"
                        ], env={**os.environ, "PYTHONPATH": "/data/openpilot"})
                elif cur_state == State.IDLE:
                    screen.set_face(FACE_STALKING)
                    if sound: sound.play(SOUND_IDLE)
            elif cur_state == State.STALKING and sound and not sound.is_playing:
                sound.play(SOUND_CHASE)
            elif cur_state == State.IDLE and sound and not sound.is_playing:
                sound.play(SOUND_IDLE)
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

            # Speed varies with distance — charge when far, creep when close
            if target_distance > 0.7:
                speed_cap = 0.45  # far — aggressive
            elif target_distance > 0.4:
                speed_cap = 0.3   # medium
            else:
                speed_cap = 0.15  # close — stalking
            final_accel = min(smooth_accel, speed_cap)
            final_steer = max(-MAX_STEER_CMD, min(MAX_STEER_CMD, smooth_bearing))

            # IDLE: slowly rotate only if we've never found anyone (initial scan)
            if cur_state == State.IDLE and not ever_seen_someone and len(faces) == 0:
                final_accel = 0.0
                final_steer = 0.05

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

            # Update occupancy grid — mark robot trail as free
            dist_moved = math.sqrt((pos_x - prev_pos_x)**2 + (pos_y - prev_pos_y)**2)
            if dist_moved > 0.05:  # moved at least 5cm
                nav_grid.mark_free_along_path(prev_pos_x, prev_pos_y, pos_x, pos_y)
                prev_pos_x = pos_x
                prev_pos_y = pos_y

            # Track commanding vs actual movement (only when driving forward, not spinning)
            if final_accel > 0.05:
                last_commanding_time = now
            if world_speed > 0.05:
                last_actually_moving_time = now

            recently_commanding = (now - last_commanding_time) < 1.0
            not_actually_moving = (now - last_actually_moving_time) > 2.0

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
                # Stuck! Mark obstacle on map in the direction we were heading
                obstacle_dist = 0.3  # obstacle is ~30cm ahead
                obs_x = pos_x + obstacle_dist * math.cos(heading)
                obs_y = pos_y + obstacle_dist * math.sin(heading)
                nav_grid.mark_obstacle(obs_x, obs_y, confidence=0.4)
                nav_follower.set_path([])  # force replan
                # Start backup sequence
                turn_direction = -1.0 if smooth_bearing > 0 else 1.0
                backup_until = now + 0.8
                turn_until = now + 2.5  # back up 0.8s then turn ~1.7s
                log.info("STUCK — backup + turn (dir=%.0f)", turn_direction)

            publisher.update(final_accel, final_steer)

            # Write state for UI eye tracking
            try:
                with open("/tmp/scp173_state", "w") as f:
                    f.write(f"{cur_state.name} {target_bearing:.3f} {time_since_face:.1f}")
            except Exception:
                pass

            elapsed = time.monotonic() - t0
            fps = 0.9 * fps + 0.1 * (1.0 / max(elapsed, 0.001))

            log.info(
                "state=%s faces=%d watched=%s bearing=%.2f dist=%.3f "
                "accel=%.2f steer=%+.2f robot=(%.2f,%.2f) person=(%.2f,%.2f) spd=%.2f fps=%.0f %dms",
                fsm.state.name, len(faces), being_watched,
                target_bearing, target_distance,
                final_accel, final_steer, pos_x, pos_y,
                person_world_x, person_world_y, world_speed,
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
        # Save occupancy grid for post-demo analysis
        try:
            np.save("/data/openpilot/scp173/occupancy_grid.npy", nav_grid.grid)
            log.info("Occupancy grid saved (%d obstacles, %d explored)",
                     nav_grid.get_obstacle_count(), nav_grid.get_explored_count())
        except Exception:
            pass


if __name__ == "__main__":
    main()
