"""SCP-173 Comma Body — tuning parameters."""

# ── Model paths ────────────────────────────────────────────────────────────────
import os
_HERE = os.path.dirname(__file__)

YOLO_MODEL_PATH    = os.path.join(_HERE, "models", "yolov8n_320.onnx")
DEPTH_MODEL_PATH   = os.path.join(_HERE, "models", "depth_anything_v2_vits.onnx")
STRIKE_SOUND_PATH  = os.path.join(_HERE, "sounds", "strike.wav")
SCRAPE_SOUND_PATH  = os.path.join(_HERE, "sounds", "scrape_loop.wav")

# ── Person detection ───────────────────────────────────────────────────────────
YOLO_INPUT_SIZE     = 320
YOLO_CONF_THRESH    = 0.5

# ── Attention detection ────────────────────────────────────────────────────────
EAR_THRESHOLD       = 0.20   # below = eyes closed → not watching
YAW_THRESHOLD_DEG   = 35.0   # |yaw| below = facing camera → watching
MAX_FACES           = 4

# ── Depth estimation ──────────────────────────────────────────────────────────
DEPTH_INPUT_SIZE    = 308

# ── State machine ─────────────────────────────────────────────────────────────
STRIKE_DISTANCE     = 0.15   # world-space distance / 3.0 — triggers at ~0.45m
STRIKE_COOLDOWN     = 3.0    # seconds in STRIKE before returning to IDLE
FREEZE_GRACE        = 0.3    # seconds of "not watched" before resuming movement

# ── Navigation / VFH ──────────────────────────────────────────────────────────
VFH_NUM_SECTORS        = 36
VFH_OBSTACLE_THRESHOLD = 0.25
VFH_ROBOT_WIDTH_SECTORS = 3
VFH_MAX_SPEED          = 0.7
VFH_TURN_GAIN          = 1.2

# ── Debug streaming ───────────────────────────────────────────────────────────
# Set STREAM_HOST to your local machine's IP to receive frames in viewer.py
# Leave empty to disable streaming
STREAM_HOST = "192.168.63.24"
STREAM_PORT = 5174

# ── Motor output ──────────────────────────────────────────────────────────────
# Published as testJoystick axes=[accel, steer] at CONTROL_HZ
CONTROL_HZ = 10     # main loop rate (Hz)

# ── Audio ──────────────────────────────────────────────────────────────────────
AUDIO_FREQ     = 44100
AUDIO_SIZE     = -16
AUDIO_CHANNELS = 1
AUDIO_BUFFER   = 512
