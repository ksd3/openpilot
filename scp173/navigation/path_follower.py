"""Pure pursuit path follower.

Given a list of waypoints and the robot's current pose, computes
a steering command to smoothly follow the path.
"""

import math


class PathFollower:
    def __init__(self, lookahead_distance: float = 0.5):
        self._lookahead = lookahead_distance
        self._path: list[tuple[float, float]] = []
        self._current_idx = 0

    def set_path(self, waypoints: list[tuple[float, float]]):
        self._path = waypoints
        self._current_idx = 0

    @property
    def has_path(self) -> bool:
        return len(self._path) > 0 and self._current_idx < len(self._path)

    @property
    def remaining_waypoints(self) -> int:
        return max(0, len(self._path) - self._current_idx)

    def get_steer(self, robot_x: float, robot_y: float, robot_heading: float,
                  fov_half: float = 1.047) -> float:
        """Compute steering command to follow the path.

        Returns bearing in [-1, 1] range (same as target_bearing in main.py).
        fov_half: half field of view in radians (default ~60°).
        """
        if not self.has_path:
            return 0.0

        # Advance past waypoints we've already reached
        while self._current_idx < len(self._path) - 1:
            wx, wy = self._path[self._current_idx]
            dist = math.sqrt((wx - robot_x) ** 2 + (wy - robot_y) ** 2)
            if dist < self._lookahead * 0.5:
                self._current_idx += 1
            else:
                break

        # Find lookahead point on the path
        lookahead_x, lookahead_y = self._find_lookahead(robot_x, robot_y)

        # Compute bearing to lookahead point
        dx = lookahead_x - robot_x
        dy = lookahead_y - robot_y
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - robot_heading

        # Normalize to [-pi, pi]
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        # Convert to [-1, 1] range
        return max(-1.0, min(1.0, angle_diff / fov_half))

    def distance_to_goal(self, robot_x: float, robot_y: float) -> float:
        """Distance from robot to final waypoint."""
        if not self._path:
            return float('inf')
        gx, gy = self._path[-1]
        return math.sqrt((gx - robot_x) ** 2 + (gy - robot_y) ** 2)

    def _find_lookahead(self, robot_x: float, robot_y: float) -> tuple[float, float]:
        """Find the point on the path that's approximately lookahead_distance ahead."""
        best_point = self._path[self._current_idx]
        best_dist_to_lookahead = float('inf')

        for i in range(self._current_idx, len(self._path)):
            wx, wy = self._path[i]
            dist = math.sqrt((wx - robot_x) ** 2 + (wy - robot_y) ** 2)
            diff = abs(dist - self._lookahead)
            if diff < best_dist_to_lookahead:
                best_dist_to_lookahead = diff
                best_point = (wx, wy)

        return best_point
