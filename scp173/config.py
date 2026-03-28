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
STRIKE_DISTANCE     = 0.001  # relative depth threshold — stops as close as possible
STRIKE_COOLDOWN     = 3.0    # seconds in STRIKE before returning to IDLE
FREEZE_GRACE        = 0.3    # seconds of "not watched" before resuming movement

# ── Navigation / VFH ──────────────────────────────────────────────────────────
VFH_NUM_SECTORS        = 36
VFH_OBSTACLE_THRESHOLD = 0.25
VFH_ROBOT_WIDTH_SECTORS = 3
VFH_MAX_SPEED          = 0.7
VFH_TURN_GAIN          = 1.2

# ── Occupancy Grid / Exploration ─────────────────────────────────────
GRID_SIZE_M            = 10.0     # physical grid extent (meters, square)
GRID_RESOLUTION        = 0.05    # meters per cell (5 cm → 200×200 grid)
GRID_CELLS             = int(GRID_SIZE_M / GRID_RESOLUTION)
DEPTH_SCALE_M          = 3.0     # depth=1.0 ≈ this many meters (tune empirically)
DEPTH_FOV_DEG          = 72.0    # wide camera horizontal FOV
LOG_ODDS_FREE          = -0.4    # free-space observation weight
LOG_ODDS_OCCUPIED      = 0.85    # obstacle observation weight
LOG_ODDS_CLAMP         = 5.0     # prevent saturation
FRONTIER_MIN_CLUSTER   = 3       # min frontier cells for valid target
EXPLORE_SPEED_SCALE    = 0.8     # scale VFH speed during exploration
GRID_RAY_SAMPLE_COLS   = 36      # depth map columns to sample per update

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
