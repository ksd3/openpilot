"""Occupancy grid mapping and frontier-based exploration."""

import math
import numpy as np

from scp173.config import (
    GRID_CELLS, GRID_RESOLUTION, GRID_SIZE_M,
    DEPTH_SCALE_M, DEPTH_FOV_DEG,
    LOG_ODDS_FREE, LOG_ODDS_OCCUPIED, LOG_ODDS_CLAMP,
    FRONTIER_MIN_CLUSTER, GRID_RAY_SAMPLE_COLS,
)


class PoseTracker:
    """Integrates cameraOdometry (trans/rot at 20 Hz) into (x, y, yaw)."""

    def __init__(self):
        self.x: float = 0.0
        self.y: float = 0.0
        self.yaw: float = 0.0
        self._last_t: float = 0.0

    def update(self, trans: list, rot: list, t: float) -> None:
        if self._last_t == 0.0:
            self._last_t = t
            return
        dt = t - self._last_t
        self._last_t = t
        if dt <= 0.0 or dt > 0.5:
            return

        forward = trans[0]
        yaw_rate = rot[2]

        self.yaw += yaw_rate * dt
        self.x += forward * dt * math.cos(self.yaw)
        self.y += forward * dt * math.sin(self.yaw)


class OccupancyGrid:
    """200x200 log-odds occupancy grid (10 m x 10 m at 5 cm resolution).

    The robot starts at the centre of the grid.  Each tick, depth-map columns
    are ray-cast into the grid to mark free and occupied cells.
    """

    def __init__(self):
        self.pose = PoseTracker()
        self._grid = np.zeros((GRID_CELLS, GRID_CELLS), dtype=np.float32)
        self._half = GRID_SIZE_M / 2.0
        self._fov_rad = math.radians(DEPTH_FOV_DEG)

    # ── public API ────────────────────────────────────────────────────

    def update(self, depth_map: np.ndarray) -> None:
        """Ray-cast sampled depth columns into the grid."""
        _, w = depth_map.shape
        mid_row = depth_map.shape[0] // 2
        cols = np.linspace(0, w - 1, GRID_RAY_SAMPLE_COLS, dtype=int)

        for col in cols:
            angle_frac = (col / w) - 0.5
            bearing = self.pose.yaw + angle_frac * self._fov_rad

            depth_val = float(depth_map[mid_row, col])
            dist_m = depth_val * DEPTH_SCALE_M
            dist_m = max(0.1, min(dist_m, GRID_SIZE_M / 2.0 - 0.1))

            self._raycast(bearing, dist_m)

    def get_exploration_bearing(self) -> float:
        """Return bearing to nearest frontier as [-1, 1].

        Frontier cells are unknown (near zero) cells adjacent to known-free
        cells.  Returns 0.5 (gentle spin) when no frontiers are found.
        """
        free_mask = self._grid < -0.5
        unknown_mask = np.abs(self._grid) < 0.1

        # Dilated adjacency: shift free_mask in 4 directions (pure numpy)
        padded = np.pad(free_mask, 1, constant_values=False)
        adjacent_to_free = (
            padded[:-2, 1:-1] | padded[2:, 1:-1] |
            padded[1:-1, :-2] | padded[1:-1, 2:]
        )

        frontier = unknown_mask & adjacent_to_free
        fy, fx = np.nonzero(frontier)

        if len(fy) < FRONTIER_MIN_CLUSTER:
            return 0.5  # fallback: gentle spin

        # Robot position in grid coords
        rx = (self.pose.x + self._half) / GRID_RESOLUTION
        ry = (self.pose.y + self._half) / GRID_RESOLUTION

        dists = (fx - rx) ** 2 + (fy - ry) ** 2
        nearest = np.argmin(dists)
        target_x = fx[nearest] * GRID_RESOLUTION - self._half
        target_y = fy[nearest] * GRID_RESOLUTION - self._half

        world_bearing = math.atan2(target_y - self.pose.y,
                                   target_x - self.pose.x)
        relative = world_bearing - self.pose.yaw
        # Normalise to [-pi, pi]
        relative = (relative + math.pi) % (2 * math.pi) - math.pi

        return float(np.clip(relative / math.pi, -1.0, 1.0))

    def get_stats(self) -> dict:
        total = GRID_CELLS * GRID_CELLS
        explored = int(np.count_nonzero(np.abs(self._grid) > 0.1))
        return {
            "pose": f"({self.pose.x:.2f},{self.pose.y:.2f})",
            "yaw_deg": round(math.degrees(self.pose.yaw) % 360, 1),
            "explored_pct": round(100.0 * explored / total, 1),
        }

    # ── internals ─────────────────────────────────────────────────────

    def _world_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        gx = int((wx + self._half) / GRID_RESOLUTION)
        gy = int((wy + self._half) / GRID_RESOLUTION)
        return (
            max(0, min(GRID_CELLS - 1, gx)),
            max(0, min(GRID_CELLS - 1, gy)),
        )

    def _raycast(self, bearing: float, dist_m: float) -> None:
        """DDA ray-cast: mark free cells along ray, occupied at endpoint."""
        ox, oy = self.pose.x, self.pose.y
        ex = ox + dist_m * math.cos(bearing)
        ey = oy + dist_m * math.sin(bearing)

        gx0, gy0 = self._world_to_grid(ox, oy)
        gx1, gy1 = self._world_to_grid(ex, ey)

        # Bresenham line
        cells = self._bresenham(gx0, gy0, gx1, gy1)

        # Free cells (all except last)
        for cx, cy in cells[:-1]:
            self._grid[cy, cx] = max(-LOG_ODDS_CLAMP,
                                     self._grid[cy, cx] + LOG_ODDS_FREE)
        # Occupied cell (last)
        if cells:
            cx, cy = cells[-1]
            self._grid[cy, cx] = min(LOG_ODDS_CLAMP,
                                     self._grid[cy, cx] + LOG_ODDS_OCCUPIED)

    @staticmethod
    def _bresenham(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return cells
