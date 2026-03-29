"""2D occupancy grid built from robot movement and collision events.

Each cell stores a probability: 0.0 = free, 0.5 = unknown, 1.0 = blocked.
The grid is centered on the robot's starting position.

Resolution: 10cm per cell. Size: 200x200 = 20m x 20m room.
"""

import numpy as np


class OccupancyGrid:
    def __init__(self, size: int = 200, resolution: float = 0.1):
        self.size = size
        self.resolution = resolution
        self.origin = size // 2  # robot starts at grid center
        self.grid = np.full((size, size), 0.5, dtype=np.float32)  # all unknown

    def world_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        gx = int(wx / self.resolution) + self.origin
        gy = int(wy / self.resolution) + self.origin
        return (np.clip(gx, 0, self.size - 1), np.clip(gy, 0, self.size - 1))

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        wx = (gx - self.origin) * self.resolution
        wy = (gy - self.origin) * self.resolution
        return (wx, wy)

    def mark_free(self, wx: float, wy: float):
        """Mark a cell as free — the robot drove through here."""
        gx, gy = self.world_to_grid(wx, wy)
        self.grid[gx, gy] = max(0.05, self.grid[gx, gy] - 0.15)

    def mark_obstacle(self, wx: float, wy: float, confidence: float = 0.4):
        """Mark a cell as occupied — got stuck or detected obstacle."""
        gx, gy = self.world_to_grid(wx, wy)
        self.grid[gx, gy] = min(0.95, self.grid[gx, gy] + confidence)
        # Also mark neighboring cells at lower confidence (obstacles have width)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = gx + dx, gy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    self.grid[nx, ny] = min(0.95, self.grid[nx, ny] + confidence * 0.5)

    def is_blocked(self, wx: float, wy: float) -> bool:
        gx, gy = self.world_to_grid(wx, wy)
        return self.grid[gx, gy] > 0.6

    def is_blocked_grid(self, gx: int, gy: int) -> bool:
        if gx < 0 or gx >= self.size or gy < 0 or gy >= self.size:
            return True  # out of bounds = blocked
        return self.grid[gx, gy] > 0.6

    def mark_free_along_path(self, x0: float, y0: float, x1: float, y1: float):
        """Mark cells as free along a line from (x0,y0) to (x1,y1)."""
        gx0, gy0 = self.world_to_grid(x0, y0)
        gx1, gy1 = self.world_to_grid(x1, y1)
        # Bresenham's line
        dx = abs(gx1 - gx0)
        dy = abs(gy1 - gy0)
        sx = 1 if gx0 < gx1 else -1
        sy = 1 if gy0 < gy1 else -1
        err = dx - dy
        steps = 0
        while steps < 200:  # safety limit
            if 0 <= gx0 < self.size and 0 <= gy0 < self.size:
                self.grid[gx0, gy0] = max(0.05, self.grid[gx0, gy0] - 0.1)
            if gx0 == gx1 and gy0 == gy1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                gx0 += sx
            if e2 < dx:
                err += dx
                gy0 += sy
            steps += 1

    def get_obstacle_count(self) -> int:
        """Number of cells marked as blocked."""
        return int(np.sum(self.grid > 0.6))

    def get_explored_count(self) -> int:
        """Number of cells that aren't unknown."""
        return int(np.sum(np.abs(self.grid - 0.5) > 0.1))
