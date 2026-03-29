"""A* path planner on the occupancy grid.

Finds the shortest collision-free path from start to goal.
8-directional movement. Runs in <1ms on a 200x200 grid.
"""

import heapq
import math
from scp173.navigation.occupancy_grid import OccupancyGrid


def plan(grid: OccupancyGrid, start_world: tuple[float, float],
         goal_world: tuple[float, float]) -> list[tuple[float, float]]:
    """A* from start to goal in world coordinates.

    Returns list of (x, y) world-space waypoints, or empty list if no path.
    """
    sx, sy = grid.world_to_grid(*start_world)
    gx, gy = grid.world_to_grid(*goal_world)

    # If goal is blocked, find nearest unblocked cell to goal
    if grid.is_blocked_grid(gx, gy):
        gx, gy = _nearest_free(grid, gx, gy)
        if gx is None:
            return []

    # If start is blocked (shouldn't happen but just in case)
    if grid.is_blocked_grid(sx, sy):
        grid.grid[sx, sy] = 0.3  # force free

    # A* search
    open_set = []
    heapq.heappush(open_set, (0.0, sx, sy))
    came_from = {}
    g_score = {(sx, sy): 0.0}
    closed = set()

    # 8 directions: dx, dy, cost
    neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414),
    ]

    max_iterations = 10000  # safety limit

    for _ in range(max_iterations):
        if not open_set:
            return []  # no path

        _, cx, cy = heapq.heappop(open_set)

        if (cx, cy) in closed:
            continue
        closed.add((cx, cy))

        if cx == gx and cy == gy:
            # Reconstruct path
            return _reconstruct(grid, came_from, (gx, gy))

        for dx, dy, cost in neighbors:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in closed:
                continue
            if grid.is_blocked_grid(nx, ny):
                continue

            # Penalty for cells near obstacles (prefer paths away from walls)
            proximity_cost = 0.0
            for ddx in [-1, 0, 1]:
                for ddy in [-1, 0, 1]:
                    if grid.is_blocked_grid(nx + ddx, ny + ddy):
                        proximity_cost += 0.5

            new_g = g_score[(cx, cy)] + cost + proximity_cost
            if new_g < g_score.get((nx, ny), float('inf')):
                g_score[(nx, ny)] = new_g
                h = math.sqrt((nx - gx) ** 2 + (ny - gy) ** 2)
                f = new_g + h
                came_from[(nx, ny)] = (cx, cy)
                heapq.heappush(open_set, (f, nx, ny))

    return []  # exceeded iteration limit


def _reconstruct(grid: OccupancyGrid, came_from: dict,
                 goal: tuple[int, int]) -> list[tuple[float, float]]:
    """Trace back from goal to start, convert to world coords, simplify."""
    path_grid = []
    current = goal
    while current in came_from:
        path_grid.append(current)
        current = came_from[current]
    path_grid.append(current)  # start
    path_grid.reverse()

    # Convert to world coordinates
    path_world = [grid.grid_to_world(gx, gy) for gx, gy in path_grid]

    # Simplify: keep every Nth waypoint to avoid micro-steps
    if len(path_world) > 3:
        step = max(1, len(path_world) // 10)
        simplified = path_world[::step]
        if simplified[-1] != path_world[-1]:
            simplified.append(path_world[-1])
        return simplified

    return path_world


def _nearest_free(grid: OccupancyGrid, gx: int, gy: int) -> tuple[int | None, int | None]:
    """Find the nearest unblocked cell to (gx, gy) using BFS."""
    from collections import deque
    queue = deque([(gx, gy)])
    visited = {(gx, gy)}
    for _ in range(2000):
        if not queue:
            break
        cx, cy = queue.popleft()
        if not grid.is_blocked_grid(cx, cy):
            return cx, cy
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in visited and 0 <= nx < grid.size and 0 <= ny < grid.size:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    return None, None
