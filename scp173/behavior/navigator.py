"""Vector Field Histogram (VFH) obstacle avoidance."""

import numpy as np

from scp173.config import (
    VFH_NUM_SECTORS, VFH_OBSTACLE_THRESHOLD, VFH_ROBOT_WIDTH_SECTORS,
    VFH_MAX_SPEED, VFH_TURN_GAIN,
)


class VFHNavigator:
    """Obstacle-aware steering using a polar obstacle histogram.

    Given a monocular depth map and a desired target bearing, returns a
    safe (speed, steer) command that avoids obstacles while heading toward
    the target.
    """

    _BLOCKED_DENSITY = 0.3   # histogram bin threshold for "blocked"

    def __init__(self):
        self._histogram = np.zeros(VFH_NUM_SECTORS)

    # ------------------------------------------------------------------
    def navigate(
        self, depth_map: np.ndarray, target_bearing: float
    ) -> tuple[float, float]:
        """Return (forward_speed 0-1, steer -1..1).

        target_bearing: -1 (hard left) … +1 (hard right)
        depth_map: (H, W) float32 normalised 0=near, 1=far
        """
        self._build_histogram(depth_map)

        target_sector = int(((target_bearing + 1.0) / 2.0) * VFH_NUM_SECTORS)
        target_sector = int(np.clip(target_sector, 0, VFH_NUM_SECTORS - 1))

        best_sector = self._best_open_sector(target_sector)

        # Sector index → normalised bearing [-1, 1]
        sector_bearing = (best_sector / VFH_NUM_SECTORS) * 2.0 - 1.0
        steer = float(np.clip(sector_bearing * VFH_TURN_GAIN, -1.0, 1.0))

        # Slow down proportionally to how blocked the front is
        half = VFH_NUM_SECTORS // 2
        front = self._histogram[half - 2: half + 3]
        front_clear = 1.0 - float(np.max(front))
        speed = VFH_MAX_SPEED * max(0.1, front_clear)

        return (speed, steer)

    # ------------------------------------------------------------------
    def _build_histogram(self, depth_map: np.ndarray) -> None:
        h, w = depth_map.shape
        # Only consider the mid-vertical band (ignore floor and ceiling)
        roi = depth_map[int(h * 0.2): int(h * 0.7), :]
        self._histogram = np.zeros(VFH_NUM_SECTORS)
        for s in range(VFH_NUM_SECTORS):
            c0 = int((s / VFH_NUM_SECTORS) * w)
            c1 = min(int(((s + 1) / VFH_NUM_SECTORS) * w), w)
            slice_ = roi[:, c0:c1]
            self._histogram[s] = float(np.mean(slice_ < VFH_OBSTACLE_THRESHOLD))

    def _best_open_sector(self, target_sector: int) -> int:
        blocked = self._histogram > self._BLOCKED_DENSITY
        half_w = VFH_ROBOT_WIDTH_SECTORS // 2
        best_sector = target_sector
        best_cost = float("inf")

        for s in range(VFH_NUM_SECTORS):
            if any(blocked[(s + off) % VFH_NUM_SECTORS]
                   for off in range(-half_w, half_w + 1)):
                continue  # not wide enough gap
            diff = abs(s - target_sector)
            diff = min(diff, VFH_NUM_SECTORS - diff)
            if diff < best_cost:
                best_cost = diff
                best_sector = s

        return best_sector
