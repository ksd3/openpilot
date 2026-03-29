"""SCP-173 face animations for the comma body dot matrix display.

Drop-in replacement for the body's default face animations.
Uses the same 8x16 grid coordinate system.
"""

from selfdrive.ui.body.animations import Animation, AnimationMode

# SCP-173 angry eyes — wide, unblinking, menacing
# Grid is roughly 8 rows x 16 columns

# Left eye — large, square-ish, aggressive
SCP_EYE_LEFT = [
(1, 1), (1, 2), (1, 3), (1, 4),
(2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
(5, 1), (5, 2), (5, 3), (5, 4),
]

# Right eye — mirror
SCP_EYE_RIGHT = [(r, 15 - c) for r, c in SCP_EYE_LEFT]

# Angry brow — angled down toward center
SCP_BROW_LEFT = [
(0, 0),
        (0, 1),
                (1, 2),
]
SCP_BROW_RIGHT = [(r, 15 - c) for r, c in SCP_BROW_LEFT]

# No mouth — SCP-173 has no visible mouth, just eyes
SCP_MOUTH = []

# Thin menacing mouth
SCP_MOUTH_THIN = [(7, 6), (7, 7), (7, 8), (7, 9)]

# Full SCP-173 face
SCP_FACE = SCP_EYE_LEFT + SCP_EYE_RIGHT + SCP_BROW_LEFT + SCP_BROW_RIGHT + SCP_MOUTH

# Narrowed eyes — when stalking (partially closed, predatory)
SCP_EYE_NARROW_LEFT = [
(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
]
SCP_EYE_NARROW_RIGHT = [(r, 15 - c) for r, c in SCP_EYE_NARROW_LEFT]

SCP_FACE_NARROW = SCP_EYE_NARROW_LEFT + SCP_EYE_NARROW_RIGHT + SCP_BROW_LEFT + SCP_BROW_RIGHT + SCP_MOUTH

# Closed — just slits
SCP_EYE_CLOSED_LEFT = [
(3, 1), (3, 2), (3, 3), (3, 4),
]
SCP_EYE_CLOSED_RIGHT = [(r, 15 - c) for r, c in SCP_EYE_CLOSED_LEFT]

SCP_FACE_CLOSED = SCP_EYE_CLOSED_LEFT + SCP_EYE_CLOSED_RIGHT


# --- Animations matching the body UI's expected format ---

# FROZEN: wide unblinking stare (replaces NORMAL)
SCP_FROZEN = Animation(
    frames=[SCP_FACE],
    mode=AnimationMode.REPEAT_FORWARD,
    frame_duration=1.0,
    repeat_interval=10.0,
)

# STALKING: narrowed predatory eyes, slow blink (replaces NORMAL)
SCP_STALKING = Animation(
    frames=[
        SCP_FACE_NARROW,
        SCP_FACE,
        SCP_FACE_NARROW,
    ],
    mode=AnimationMode.REPEAT_FORWARD_BACKWARD,
    frame_duration=0.5,
    repeat_interval=3.0,
)

# IDLE: eyes slowly open and close (replaces ASLEEP)
SCP_IDLE = Animation(
    frames=[
        SCP_FACE_CLOSED,
        SCP_EYE_NARROW_LEFT + SCP_EYE_NARROW_RIGHT,
        SCP_FACE,
        SCP_EYE_NARROW_LEFT + SCP_EYE_NARROW_RIGHT,
        SCP_FACE_CLOSED,
    ],
    mode=AnimationMode.REPEAT_FORWARD,
    frame_duration=0.4,
    repeat_interval=5.0,
)

# STRIKE: rapid flash
SCP_STRIKE = Animation(
    frames=[
        SCP_FACE,
        [],  # blank
        SCP_FACE,
        [],
        SCP_FACE,
    ],
    mode=AnimationMode.ONCE_FORWARD,
    frame_duration=0.1,
)
