"""SCP-173 four-state finite state machine."""

import time
from enum import Enum, auto

from scp173.config import STRIKE_DISTANCE, STRIKE_COOLDOWN, FREEZE_GRACE


class State(Enum):
    IDLE     = auto()
    STALKING = auto()
    FROZEN   = auto()
    STRIKE   = auto()


class SCP173StateMachine:
    """Behavioural core of the SCP-173 robot.

    Call update() every control tick.  Returns (accel, steer) motor command
    in the range [0, 1] and [-1, 1] respectively.

    State transitions:
      IDLE → STALKING  : person detected and not being watched
      STALKING → FROZEN: being watched (instant freeze)
      STALKING → STRIKE: person within STRIKE_DISTANCE while not watched
      FROZEN → STALKING: not watched for longer than FREEZE_GRACE
      STRIKE → IDLE    : after STRIKE_COOLDOWN seconds
    """

    def __init__(self, on_strike_callback=None):
        self.state: State = State.IDLE
        self._strike_time: float = 0.0
        self._last_watched_time: float = 0.0
        self._on_strike = on_strike_callback  # called once on entering STRIKE

    # ------------------------------------------------------------------
    def update(
        self,
        person_detected: bool,
        being_watched: bool,
        target_bearing: float,   # -1 (left) … +1 (right)
        target_distance: float,  # relative depth, 0 = right here
        now: float | None = None,
    ) -> tuple[float, float]:
        """Return (accel 0-1, steer -1..1) for this tick."""
        t = now if now is not None else time.monotonic()

        if self.state == State.IDLE:
            if person_detected and not being_watched:
                self.state = State.STALKING
                # Fall through to STALKING logic immediately
            else:
                return (0.0, 0.0)

        if self.state == State.STALKING:
            if being_watched:
                self.state = State.FROZEN
                self._last_watched_time = t
                return (0.0, 0.0)

            if not person_detected:
                self.state = State.IDLE
                return (0.0, 0.0)

            if target_distance < STRIKE_DISTANCE:
                self.state = State.STRIKE
                self._strike_time = t
                if self._on_strike:
                    self._on_strike()
                return (0.0, 0.0)

            # Advance toward target
            accel = min(0.6, target_distance * 1.5)
            steer = target_bearing * 0.8
            return (accel, float(np.clip(steer, -1.0, 1.0)))

        elif self.state == State.FROZEN:
            if being_watched:
                self._last_watched_time = t
                return (0.0, 0.0)
            if (t - self._last_watched_time) > FREEZE_GRACE:
                self.state = State.STALKING
            return (0.0, 0.0)

        elif self.state == State.STRIKE:
            if (t - self._strike_time) > STRIKE_COOLDOWN:
                self.state = State.IDLE
            return (0.0, 0.0)

        return (0.0, 0.0)


# numpy is only needed for the clip in STALKING; import lazily so the module
# can still be imported even without numpy (e.g. unit tests that mock it)
try:
    import numpy as np
except ImportError:
    class np:  # type: ignore
        @staticmethod
        def clip(v, lo, hi):
            return max(lo, min(hi, v))
