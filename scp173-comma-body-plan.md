# SCP-173 Comma Body — Full Technical Plan

## Executive summary

Turn a Comma 4 + Custom Body 3 into an indoor robot that stalks humans when
they aren't looking, freezes when observed, avoids obstacles using monocular
depth estimation, and plays a sound effect on "capture." All perception runs
on the Comma 4's Snapdragon 845 MAX — no external depth sensors.

---

## 1. Hardware inventory

| Component | Role | Notes |
|---|---|---|
| **Comma 4** | Brain — Snapdragon 845 MAX, 3 cameras, IMU, Wi-Fi/LTE, 1.9" OLED | Runs Ubuntu, SSH-accessible, full `apt-get` |
| **Custom Body 3** | Wheeled chassis, motors, battery | Must expose CAN bus or serial motor interface |
| *(Comma 4 built-in speaker)* | Sound effects | Already on-board — no extra hardware needed |

### Camera layout on the Body

When the Comma 4 sits atop the body platform:

- **Road camera** (forward-facing, narrow FOV): Primary for *person detection* and *target tracking*
- **Wide road camera** (forward-facing, wide FOV ~120°): Primary for *obstacle detection* and *depth estimation* — the wide FOV sees furniture legs, walls, and floor obstacles the narrow cam misses
- **Driver camera** (originally rear-facing toward the driver): On the body this points *backward* or *downward* depending on mount angle — useful for rear obstacle detection or as a secondary person-detector (catches people sneaking up behind)

**Recommendation:** Mount the Comma 4 so the road + wide cams face forward and the driver cam faces backward. This gives you near-360° awareness.

---

## 2. Software stack

### Runtime environment

```
OS:           Ubuntu (on Comma 4, via SSH)
Language:     Python 3.10+
Control API:  bodyjim (gymnasium-like API) OR custom CAN interface
ML runtime:   ONNX Runtime (CPU) or Qualcomm SNPE (GPU/DSP)
```

### Core dependencies

```bash
# On the Comma 4 (SSH in)
pip install opencv-python-headless numpy onnxruntime mediapipe pygame

# For body control (if using bodyjim)
pip install bodyjim

# For depth estimation model
pip install torch torchvision  # only if converting models; inference uses ONNX
```

### Key libraries & models

| Purpose | Library / Model | Why this one |
|---|---|---|
| Person detection | **YOLOv8-nano (ONNX)** | 3.2M params, runs at ~30 FPS on CPU, accurate for people at indoor distances (1-8m) |
| Face mesh + eye state | **MediaPipe Face Mesh** | 468 landmarks, real-time on mobile SoCs, gives eye aspect ratio (EAR) for blink detection and head pose for gaze direction — no extra model needed |
| Monocular depth | **Depth Anything V2 — Small** | State-of-the-art indoor depth estimation, trained on NYU Depth V2 (indoor dataset), runs at ~15 FPS on ONNX with the small encoder |
| Motor control | **bodyjim** or custom CAN | Gymnasium-style `env.step((x, y))` API; if your custom Body 3 has a different interface, wrap it in this pattern |
| Audio | **pygame.mixer** | Lightweight, supports WAV/OGG, low latency |

---

## 3. Perception pipeline — detailed breakdown

### 3A. Person detection (YOLOv8-nano)

**What it does:** Detects bounding boxes of people in the camera frame.

**Setup:**
```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=416
# This gives you yolov8n.onnx — copy to the Comma 4
```

**Integration pattern:**
```python
import onnxruntime as ort
import cv2
import numpy as np

class PersonDetector:
    def __init__(self, model_path="yolov8n.onnx", conf_thresh=0.5):
        self.session = ort.InferenceSession(model_path)
        self.conf_thresh = conf_thresh
        self.input_size = 416

    def detect(self, frame):
        """Returns list of (x1, y1, x2, y2, confidence) for people only."""
        blob = cv2.resize(frame, (self.input_size, self.input_size))
        blob = blob.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis]

        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        # Parse YOLO output, filter for class 0 (person), apply NMS
        # ... (standard YOLO post-processing)
        return people_boxes
```

**Performance budget:** ~15-20ms per frame at 416×416 on Snapdragon 845 CPU.

### 3B. Attention detection (the SCP-173 core mechanic)

This is the hardest and most important subsystem. You need to answer:
**"Is anyone currently looking at me?"**

**Approach: Head pose + Eye Aspect Ratio (EAR)**

Rather than full gaze tracking (which requires calibration and is brittle), use two
simpler signals that combine into a reliable "being watched" detector:

1. **Head pose estimation** — If the person's face is oriented toward the camera
   (within ±30° yaw), they *could* be watching.
2. **Eye Aspect Ratio (EAR)** — If EAR drops below threshold (~0.2), eyes are
   closed → the person is NOT watching regardless of head direction.

```python
import mediapipe as mp
import numpy as np

class AttentionDetector:
    """Determines if any detected person is actively watching the robot."""

    # MediaPipe face mesh landmark indices for eye contours
    LEFT_EYE = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    # Landmark indices for head pose estimation (nose tip, chin, etc.)
    POSE_LANDMARKS = [1, 33, 263, 61, 291, 199]

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def eye_aspect_ratio(self, landmarks, eye_indices):
        """Compute EAR: ratio of eye height to width.
        Low EAR = closed eyes. Threshold ~0.2."""
        pts = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
        # Vertical distances
        v1 = np.linalg.norm(pts[1] - pts[5])
        v2 = np.linalg.norm(pts[2] - pts[4])
        # Horizontal distance
        h = np.linalg.norm(pts[0] - pts[3])
        return (v1 + v2) / (2.0 * h)

    def estimate_head_yaw(self, landmarks, frame_shape):
        """Estimate head yaw angle using solvePnP.
        Returns yaw in degrees. |yaw| < 30 = facing camera."""
        h, w = frame_shape[:2]
        # 2D image points
        image_pts = np.array([
            (landmarks[i].x * w, landmarks[i].y * h)
            for i in self.POSE_LANDMARKS
        ], dtype=np.float64)

        # 3D model points (generic face model)
        model_pts = np.array([
            (0.0, 0.0, 0.0),        # Nose tip
            (-225.0, 170.0, -135.0), # Left eye corner
            (225.0, 170.0, -135.0),  # Right eye corner
            (-150.0, -150.0, -125.0),# Left mouth corner
            (150.0, -150.0, -125.0), # Right mouth corner
            (0.0, -330.0, -65.0)     # Chin
        ], dtype=np.float64)

        focal_length = w
        camera_matrix = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        _, rvec, _ = cv2.solvePnP(
            model_pts, image_pts, camera_matrix,
            np.zeros((4, 1), dtype=np.float64)
        )
        rmat, _ = cv2.Rodrigues(rvec)
        # Extract yaw (rotation around Y axis)
        yaw = np.degrees(np.arctan2(rmat[0, 2], rmat[2, 2]))
        return yaw

    def is_being_watched(self, frame):
        """Main method. Returns (bool, num_people_detected).

        True = at least one person is looking at the robot with open eyes.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return False, 0

        for face_landmarks in results.multi_face_landmarks:
            lm = face_landmarks.landmark

            # Check eye state
            left_ear = self.eye_aspect_ratio(lm, self.LEFT_EYE)
            right_ear = self.eye_aspect_ratio(lm, self.RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < 0.2:
                # Eyes closed — not watching
                continue

            # Check head orientation
            yaw = self.estimate_head_yaw(lm, frame.shape)

            if abs(yaw) < 35:
                # Facing the camera with open eyes = WATCHING
                return True, len(results.multi_face_landmarks)

        # People detected but none are watching
        return False, len(results.multi_face_landmarks)
```

**Key tuning parameters:**
- `EAR threshold` (0.2): Lower = only triggers on very closed eyes. Test with your lighting.
- `Yaw threshold` (35°): How far turned away before you consider them "not looking." 30-40° works for most indoor scenarios.
- `min_detection_confidence` (0.5): Lower if faces aren't detected at distance. At 5m+ faces will be small — consider switching to person-detection-only mode at far range.

**Edge cases to handle:**
- Person wearing sunglasses → EAR unreliable → fall back to head pose only
- Person in profile → yaw > 35° → not watching → robot moves
- Multiple people → ANY person watching = FROZEN (conservative)
- No face detected but person body detected → assume NOT watching (they're turned away) → robot moves

### 3C. Monocular depth estimation (Depth Anything V2)

**What it does:** Generates a per-pixel depth map from a single RGB frame. This replaces
a physical depth sensor (LiDAR, RealSense).

**Model choice:** Depth Anything V2 — Small encoder

- Trained on massive indoor+outdoor datasets including NYU Depth V2
- The "small" variant balances accuracy with inference speed
- Produces *relative* depth (not metric) — good enough for obstacle avoidance
  since we care about "is something close?" not "is it exactly 1.3m away?"

**Setup:**
```bash
# Clone and export to ONNX
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
python export_onnx.py --encoder vits --input-size 308  # small encoder
# Produces depth_anything_v2_vits.onnx
```

**Integration:**
```python
import onnxruntime as ort
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self, model_path="depth_anything_v2_vits.onnx"):
        self.session = ort.InferenceSession(model_path)
        self.input_size = 308

    def estimate(self, frame):
        """Returns depth map (H, W) where higher = farther."""
        img = cv2.resize(frame, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        depth = self.session.run(None, {
            self.session.get_inputs()[0].name: img
        })[0][0]

        # Normalize to 0-1 range
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        return depth  # shape (308, 308), 0=near, 1=far

    def get_obstacle_map(self, depth, near_threshold=0.3):
        """Convert depth map to binary obstacle map.
        Returns mask where True = obstacle (too close)."""
        # Lower third of frame = floor/ground plane — ignore it
        h = depth.shape[0]
        obstacle_region = depth[:int(h * 0.75), :]  # top 75% of frame
        obstacles = obstacle_region < near_threshold
        return obstacles
```

**Performance budget:** ~50-70ms per frame at 308×308 on Snapdragon 845 CPU.
You can drop to 224×224 for ~35ms if needed.

**Critical insight for monocular depth indoors:** Relative depth is fine for obstacle
avoidance because you only need to answer "is this region closer than that region?"
The depth map reliably shows furniture legs, chair backs, table edges, walls, and
doorframes as closer (darker) regions. You don't need metric depth to navigate.

---

## 4. The SCP-173 state machine

This is the behavioral core. Four states, clean transitions:

```
                    ┌──────────────────────────┐
                    │                          │
                    ▼                          │
    ┌──────┐   target    ┌──────────┐   being    ┌────────┐
    │ IDLE │──detected──▶│ STALKING │──watched──▶│ FROZEN │
    └──────┘             └──────────┘            └────────┘
                              │                      │
                         close enough           not watched
                              │                      │
                              ▼                      │
                         ┌─────────┐                 │
                         │ STRIKE  │                 │
                         │(sound!) │                 │
                         └─────────┘                 │
                              │                      │
                         after cooldown              │
                              │                      │
                              ▼                      │
                           (back to IDLE)            │
                                                     │
                    ┌────────────────────────────────┘
                    │ (back to STALKING)
                    ▼
```

```python
from enum import Enum, auto
import time
import pygame

class State(Enum):
    IDLE = auto()
    STALKING = auto()
    FROZEN = auto()
    STRIKE = auto()

class SCP173StateMachine:
    STRIKE_DISTANCE = 0.15    # depth threshold — "close enough"
    STRIKE_COOLDOWN = 3.0     # seconds before returning to IDLE after strike
    FREEZE_GRACE = 0.3        # seconds of "not watched" before resuming movement
                               # (prevents jitter from detection flicker)

    def __init__(self, sound_path="scp173_sound.wav"):
        pygame.mixer.init()
        self.sound = pygame.mixer.Sound(sound_path)
        self.state = State.IDLE
        self.strike_time = 0
        self.last_watched_time = 0
        self.target_bearing = 0.0   # -1.0 (left) to 1.0 (right)
        self.target_distance = 1.0  # relative depth, 0=touching

    def update(self, person_detected, being_watched, target_bearing,
               target_distance, current_time=None):
        """Call every frame. Returns (motor_x, motor_y) command."""
        t = current_time or time.time()

        if self.state == State.IDLE:
            if person_detected and not being_watched:
                self.state = State.STALKING
                self.target_bearing = target_bearing
                self.target_distance = target_distance
            return (0.0, 0.0)  # stationary

        elif self.state == State.STALKING:
            if being_watched:
                self.state = State.FROZEN
                self.last_watched_time = t
                return (0.0, 0.0)  # instant freeze

            if not person_detected:
                self.state = State.IDLE
                return (0.0, 0.0)

            self.target_bearing = target_bearing
            self.target_distance = target_distance

            if target_distance < self.STRIKE_DISTANCE:
                self.state = State.STRIKE
                self.strike_time = t
                self.sound.play()
                return (0.0, 0.0)

            # Move toward target
            forward_speed = min(0.6, target_distance * 1.5)
            turn = target_bearing * 0.8
            return (forward_speed, turn)

        elif self.state == State.FROZEN:
            if being_watched:
                self.last_watched_time = t
                return (0.0, 0.0)  # stay frozen

            # Grace period — don't resume instantly
            if (t - self.last_watched_time) > self.FREEZE_GRACE:
                self.state = State.STALKING
            return (0.0, 0.0)

        elif self.state == State.STRIKE:
            if (t - self.strike_time) > self.STRIKE_COOLDOWN:
                self.state = State.IDLE
            return (0.0, 0.0)  # stationary during strike
```

---

## 5. Navigation & obstacle avoidance

You can't just drive straight at someone — there are tables, chairs, and walls in
the way. The navigation system takes the depth map and the desired target
direction, and outputs a safe motor command.

### Algorithm: Vector Field Histogram (VFH)

VFH is the sweet spot for this project — more capable than simple reactive
avoidance, much less complex than full SLAM. It works by:

1. Converting the depth map into a **polar obstacle histogram** (angular sectors around the robot)
2. Finding **open valleys** (contiguous sectors with no obstacles)
3. Choosing the valley closest to the target direction
4. Steering toward the center of that valley

```python
import numpy as np

class VFHNavigator:
    """Vector Field Histogram for obstacle avoidance."""

    NUM_SECTORS = 36          # 10° per sector
    OBSTACLE_THRESHOLD = 0.25 # depth < this = blocked
    ROBOT_WIDTH_SECTORS = 3   # robot needs 3 sectors (~30°) of clearance
    MAX_SPEED = 0.7
    TURN_GAIN = 1.2

    def __init__(self):
        self.histogram = np.zeros(self.NUM_SECTORS)

    def depth_to_histogram(self, depth_map):
        """Convert depth map to polar obstacle density histogram."""
        h, w = depth_map.shape
        self.histogram = np.zeros(self.NUM_SECTORS)

        # Only look at the middle vertical band (not floor/ceiling)
        roi = depth_map[int(h*0.2):int(h*0.7), :]

        for sector in range(self.NUM_SECTORS):
            # Map sector to horizontal pixel range
            angle = (sector / self.NUM_SECTORS) * 2 * np.pi - np.pi
            col_start = int((sector / self.NUM_SECTORS) * w)
            col_end = int(((sector + 1) / self.NUM_SECTORS) * w)
            col_end = min(col_end, w)

            sector_slice = roi[:, col_start:col_end]
            # Count near-obstacles as density
            obstacle_density = np.mean(sector_slice < self.OBSTACLE_THRESHOLD)
            self.histogram[sector] = obstacle_density

    def find_best_direction(self, target_sector):
        """Find the open valley closest to the target direction."""
        # Threshold histogram into blocked/free
        blocked = self.histogram > 0.3

        # Find contiguous free valleys wide enough for the robot
        best_sector = target_sector
        best_cost = float('inf')

        for s in range(self.NUM_SECTORS):
            # Check if a window of ROBOT_WIDTH_SECTORS centered on s is clear
            clear = True
            for offset in range(-self.ROBOT_WIDTH_SECTORS // 2,
                                 self.ROBOT_WIDTH_SECTORS // 2 + 1):
                idx = (s + offset) % self.NUM_SECTORS
                if blocked[idx]:
                    clear = False
                    break

            if clear:
                # Cost = angular distance to target
                diff = abs(s - target_sector)
                diff = min(diff, self.NUM_SECTORS - diff)
                if diff < best_cost:
                    best_cost = diff
                    best_sector = s

        return best_sector

    def navigate(self, depth_map, target_bearing):
        """Main method.
        target_bearing: -1.0 (left) to 1.0 (right)
        Returns: (forward_speed, turn_rate)
        """
        self.depth_to_histogram(depth_map)

        # Convert bearing to sector
        target_sector = int(((target_bearing + 1.0) / 2.0) * self.NUM_SECTORS)
        target_sector = np.clip(target_sector, 0, self.NUM_SECTORS - 1)

        best_sector = self.find_best_direction(target_sector)

        # Convert back to turn command
        sector_bearing = (best_sector / self.NUM_SECTORS) * 2.0 - 1.0
        turn = sector_bearing * self.TURN_GAIN

        # Slow down if obstacles are close ahead
        front_sectors = self.histogram[
            self.NUM_SECTORS//2 - 2 : self.NUM_SECTORS//2 + 3
        ]
        front_clear = 1.0 - np.max(front_sectors)
        speed = self.MAX_SPEED * max(0.1, front_clear)

        return (speed, np.clip(turn, -1.0, 1.0))
```

### Why VFH over other approaches

| Approach | Pros | Cons | Verdict |
|---|---|---|---|
| **Bug algorithm** | Dead simple | Gets stuck in concave spaces, no look-ahead | Too primitive |
| **Potential fields** | Smooth paths | Local minima (robot oscillates between two chairs) | Frustrating to tune |
| **VFH** | Handles corridors, doorways, furniture clusters; no map needed | Slightly more code | **Best fit** |
| **Full SLAM + A*** | Globally optimal paths | Needs metric depth, heavy compute, overkill for "stalk the nearest human" | Over-engineered |
| **RL-based** | Could learn great policies | Needs simulation, training, lots of data | Future upgrade |

---

## 6. Motor control interface

### If using bodyjim (for standard Comma Body-compatible hardware)

```python
from bodyjim import BodyEnv

class MotorController:
    def __init__(self, body_ip):
        self.env = BodyEnv(
            body_ip,
            cameras=["road", "wideRoad", "driver"],
            services=["accelerometer", "gyroscope"]
        )

    def send_command(self, forward, turn):
        """forward: 0-1, turn: -1 (left) to 1 (right)"""
        obs, _, _, _, _ = self.env.step((forward, turn))
        return obs

    def get_frames(self, obs):
        """Extract camera frames from observation."""
        road = obs["cameras"]["road"]
        wide = obs["cameras"]["wideRoad"]
        driver = obs["cameras"]["driver"]
        return road, wide, driver
```

### If your Custom Body 3 has a different interface

Wrap whatever serial/CAN/GPIO protocol your body uses in the same
`send_command(forward, turn)` pattern. The rest of the stack doesn't
care what's underneath.

```python
import can  # python-can library

class CustomBody3Controller:
    def __init__(self, can_interface="can0"):
        self.bus = can.interface.Bus(can_interface, bustype='socketcan')

    def send_command(self, forward, turn):
        """Translate high-level commands to CAN messages for your motors."""
        # Map forward/turn to left_motor/right_motor speeds
        # (differential drive)
        left_speed = forward + turn
        right_speed = forward - turn
        left_speed = int(np.clip(left_speed, -1, 1) * 1000)
        right_speed = int(np.clip(right_speed, -1, 1) * 1000)

        # Send CAN messages (your Body 3's protocol)
        msg = can.Message(
            arbitration_id=0x200,
            data=struct.pack('>hh', left_speed, right_speed),
            is_extended_id=False
        )
        self.bus.send(msg)
```

---

## 7. Main loop — putting it all together

```python
#!/usr/bin/env python3
"""SCP-173 Comma Body — Main control loop."""

import time
import cv2
from person_detector import PersonDetector
from attention_detector import AttentionDetector
from depth_estimator import DepthEstimator
from state_machine import SCP173StateMachine
from navigator import VFHNavigator
from motor_controller import MotorController  # or CustomBody3Controller

# ── INIT ──────────────────────────────────────────────
BODY_IP = "192.168.1.100"  # your body's IP
FPS_TARGET = 10            # 10 Hz control loop is plenty

detector = PersonDetector("models/yolov8n.onnx")
attention = AttentionDetector()
depth = DepthEstimator("models/depth_anything_v2_vits.onnx")
state_machine = SCP173StateMachine("sounds/scp173_strike.wav")
navigator = VFHNavigator()
motor = MotorController(BODY_IP)

print("SCP-173 online. Beginning containment breach...")

# ── MAIN LOOP ─────────────────────────────────────────
while True:
    loop_start = time.time()

    # 1. Get camera frames
    obs = motor.send_command(0, 0)  # no-op to get fresh frames
    road_frame, wide_frame, driver_frame = motor.get_frames(obs)

    # 2. Detect people (use road cam — better resolution for faces)
    people = detector.detect(road_frame)

    person_detected = len(people) > 0
    target_bearing = 0.0
    target_distance = 1.0

    if person_detected:
        # Track the closest/largest person
        best = max(people, key=lambda p: (p[2]-p[0]) * (p[3]-p[1]))
        x1, y1, x2, y2, conf = best
        frame_w = road_frame.shape[1]

        # Bearing: center of bbox relative to frame center
        center_x = (x1 + x2) / 2.0
        target_bearing = (center_x / frame_w) * 2.0 - 1.0  # -1 to 1

        # Distance: use depth at person's bbox center
        depth_map = depth.estimate(wide_frame)
        # Map person bbox to depth map coordinates
        dx = int((center_x / frame_w) * depth_map.shape[1])
        dy = int(((y1 + y2) / 2.0 / road_frame.shape[0]) * depth_map.shape[0])
        dx = max(0, min(dx, depth_map.shape[1] - 1))
        dy = max(0, min(dy, depth_map.shape[0] - 1))
        target_distance = depth_map[dy, dx]

    # 3. Check attention state
    being_watched, num_faces = attention.is_being_watched(road_frame)

    # Also check driver cam (rear) for people watching from behind
    rear_watched, _ = attention.is_being_watched(driver_frame)
    being_watched = being_watched or rear_watched

    # 4. Run state machine
    raw_cmd = state_machine.update(
        person_detected, being_watched,
        target_bearing, target_distance
    )

    # 5. If moving, run through obstacle avoidance
    if raw_cmd[0] > 0.05:  # only navigate if we're supposed to move
        depth_map = depth.estimate(wide_frame)
        cmd = navigator.navigate(depth_map, raw_cmd[1])
        # Blend: use navigator's turn but state machine's speed intent
        final_cmd = (min(raw_cmd[0], cmd[0]), cmd[1])
    else:
        final_cmd = (0.0, 0.0)

    # 6. Send motor command
    motor.send_command(final_cmd[0], final_cmd[1])

    # 7. Rate limit
    elapsed = time.time() - loop_start
    sleep_time = max(0, (1.0 / FPS_TARGET) - elapsed)
    time.sleep(sleep_time)

    # Debug output
    print(f"State: {state_machine.state.name:10s} | "
          f"People: {len(people)} | Watched: {being_watched} | "
          f"Cmd: ({final_cmd[0]:.2f}, {final_cmd[1]:.2f}) | "
          f"Loop: {(elapsed*1000):.0f}ms")
```

---

## 8. Performance budget & optimization

### Per-frame timing breakdown (Snapdragon 845, CPU, ONNX Runtime)

| Step | Time (ms) | Resolution | Run every frame? |
|---|---|---|---|
| YOLOv8-nano | 15-20 | 416×416 | Yes |
| MediaPipe Face Mesh | 8-12 | Native | Yes (if person detected) |
| Depth Anything V2 Small | 50-70 | 308×308 | Only when moving |
| VFH computation | 1-2 | N/A | Only when moving |
| **Total (moving)** | **75-105** | — | ~10 FPS |
| **Total (frozen/idle)** | **25-35** | — | ~30 FPS |

### Optimization strategies (in order of impact)

1. **Run depth only when STALKING** — saves 50-70ms/frame in other states
2. **Alternate person detection and depth** — odd frames: detect people, even frames: estimate depth. Halves GPU load, still 5 Hz each.
3. **Quantize models to INT8** — ONNX Runtime supports quantization; can cut inference time 30-40%
4. **Use Qualcomm SNPE** — the Snapdragon 845 has a Hexagon DSP that can run quantized models 2-3x faster than CPU. Requires converting ONNX → DLC format. This is the biggest win but takes more setup.
5. **Reduce depth resolution** — 224×224 instead of 308×308 cuts depth inference to ~35ms with acceptable quality for obstacle avoidance.

---

## 9. Sound system

### Setup

```bash
# Test audio on the Comma 4's built-in speaker
aplay -l  # list audio devices — the onboard speaker should appear as a card
aplay /usr/share/sounds/alsa/Front_Center.wav  # quick test

# If the speaker doesn't output, check ALSA config:
cat /proc/asound/cards
# You may need to set the default card in ~/.asoundrc

# Install pygame for programmatic control
pip install pygame
```

### Sound effect recommendations

The Comma 4's built-in speaker is small (designed for navigation alerts), so keep
these tips in mind:
- Choose **mid-frequency sounds** (500 Hz–4 kHz) — tiny speakers roll off bass heavily
- Normalize your WAV/OGG files to **-3 dBFS** so they're loud without clipping
- If you find the speaker too quiet for your space, you can always add a USB or
  Bluetooth speaker later — the code stays identical

- **Approach sound:** Low rumble or scraping sound (concrete on concrete, per SCP lore) — play in a loop during STALKING state at low volume, increasing as distance decreases. Boost the mid-range frequencies so it cuts through on the small speaker.
- **Strike sound:** Sharp, loud stinger (stone snap, neck crack sound effect) — play once on entering STRIKE state. High-mid frequencies work best here.
- **Ambient sound:** Optional heartbeat or breathing sound during IDLE

```python
import pygame

class AudioEngine:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
        self.strike_sound = pygame.mixer.Sound("sounds/strike.wav")
        self.approach_sound = pygame.mixer.Sound("sounds/scrape_loop.wav")
        self.approach_channel = pygame.mixer.Channel(0)
        self.strike_channel = pygame.mixer.Channel(1)

    def update(self, state, distance=1.0):
        if state == State.STALKING:
            if not self.approach_channel.get_busy():
                self.approach_channel.play(self.approach_sound, loops=-1)
            # Volume increases as distance decreases
            vol = max(0.1, 1.0 - distance)
            self.approach_channel.set_volume(vol)
        elif state == State.STRIKE:
            self.approach_channel.stop()
            if not self.strike_channel.get_busy():
                self.strike_channel.play(self.strike_sound)
        else:
            self.approach_channel.fadeout(500)
```

---

## 10. Project structure

```
scp173-comma-body/
├── main.py                    # Main control loop
├── config.py                  # Tuning parameters
├── perception/
│   ├── person_detector.py     # YOLOv8-nano wrapper
│   ├── attention_detector.py  # MediaPipe head pose + EAR
│   └── depth_estimator.py     # Depth Anything V2 wrapper
├── behavior/
│   ├── state_machine.py       # SCP-173 FSM
│   └── navigator.py           # VFH obstacle avoidance
├── control/
│   ├── motor_controller.py    # bodyjim or custom CAN interface
│   └── audio_engine.py        # Sound effect system
├── models/
│   ├── yolov8n.onnx           # Person detection model
│   └── depth_anything_v2_vits.onnx  # Depth estimation model
├── sounds/
│   ├── strike.wav
│   └── scrape_loop.wav
└── tools/
    ├── export_yolo.py         # Script to export YOLOv8 to ONNX
    ├── export_depth.py        # Script to export Depth Anything to ONNX
    └── test_cameras.py        # Camera feed test utility
```

---

## 11. Development roadmap

### Phase 1 — Basic movement (Week 1-2)
- [ ] SSH into Comma 4, verify camera feeds work (`tools/test_cameras.py`)
- [ ] Get motor control working (bodyjim or custom CAN)
- [ ] Drive the body around via keyboard/gamepad over SSH
- [ ] Verify IMU and odometry data

### Phase 2 — Person detection (Week 2-3)
- [ ] Export YOLOv8-nano to ONNX, deploy to Comma 4
- [ ] Run person detection on road camera, verify bounding boxes
- [ ] Implement target tracking (largest/closest person selection)
- [ ] Basic "drive toward the detected person" behavior

### Phase 3 — Attention detection (Week 3-4)
- [ ] Integrate MediaPipe Face Mesh
- [ ] Implement EAR for blink/eye-closed detection
- [ ] Implement head pose estimation for gaze direction
- [ ] Build and test the `is_being_watched()` method
- [ ] Tune thresholds with real people in your space

### Phase 4 — SCP-173 state machine (Week 4-5)
- [ ] Implement the four-state FSM
- [ ] Wire perception outputs into state machine inputs
- [ ] Test the freeze/move behavior with a human
- [ ] Add the grace period (FREEZE_GRACE) to prevent jitter
- [ ] Add sound effects on state transitions

### Phase 5 — Obstacle avoidance (Week 5-6)
- [ ] Deploy Depth Anything V2 to Comma 4
- [ ] Implement VFH navigator
- [ ] Test navigation around chairs, tables, and doorways
- [ ] Tune obstacle threshold and sector resolution
- [ ] Integrate navigation with the state machine

### Phase 6 — Polish (Week 6-8)
- [ ] Optimize inference (INT8 quantization, SNPE if ambitious)
- [ ] Add rear-camera person detection (driver cam)
- [ ] Tune all parameters in your specific indoor environment
- [ ] Add approach sound that scales with distance
- [ ] Handle edge cases (sunglasses, multiple rooms, losing target)
- [ ] Add a physical kill switch (always have a kill switch on robots)

---

## 12. Key risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **Face detection fails at >5m** | Robot never transitions to STALKING | Fall back to body-only detection; assume "not watching" if no face visible |
| **Monocular depth fails on glass/mirrors** | Robot crashes into glass tables | Add ultrasonic sensor ($5) as a contact-range backup |
| **Snapdragon 845 too slow for all three models** | Low FPS, jerky behavior | Alternate model inference across frames; reduce resolutions |
| **Bodyjim API incompatible with Custom Body 3** | Can't control motors | Write a thin CAN/serial adapter; the API contract is just `(forward, turn)` |
| **Robot tips over** | Hardware damage | Keep speeds moderate (MAX_SPEED = 0.7); the Body's self-balancing helps |
| **People are scared** | Social consequences | Add a visible LED indicator showing state; give people a verbal warning it's a game |

---

## 13. Future upgrades

- **SLAM integration**: openpilot's localizer is heading toward 6-DOF — once
  available, add persistent mapping so the robot can navigate room-to-room
- **Multi-person priority**: Instead of "freeze if ANYONE watches," implement
  occlusion-aware targeting — move when ALL observers look away
- **Reinforcement learning**: Train a policy in simulation (Isaac Gym) that
  learns optimal stalking behavior, then deploy to the real robot
- **Arm attachment**: When Comma ships the Body arm module, add a physical
  "tag" mechanism on STRIKE instead of just sound
- **Voice interaction**: Play SCP-themed audio warnings ("SCP-173 has breached
  containment") through the built-in speaker, or add a louder USB speaker for
  bigger spaces
