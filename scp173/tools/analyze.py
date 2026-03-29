#!/usr/bin/env python3
"""SCP-173 Session Analyzer — parse logs and generate presentation visuals.

Usage:
  # Copy logs from device:
  scp comma@192.168.63.120:/data/openpilot/scp173/scp173.log ./scp173.log
  scp comma@192.168.63.120:/data/openpilot/scp173/occupancy_grid.npy ./occupancy_grid.npy

  # Generate visuals:
  python scp173/tools/analyze.py scp173.log --grid occupancy_grid.npy --output ./analysis/
"""

import argparse
import os
import re
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
except ImportError:
    print("pip install matplotlib")
    exit(1)


def parse_log(path: str) -> list[dict]:
    """Parse scp173.log into a list of frame records."""
    pattern = re.compile(
        r'(\d+:\d+:\d+) state=(\w+) faces=(\d+) watched=(\w+) .*?'
        r'bearing=([-\d.]+) dist=([-\d.]+) '
        r'accel=([-\d.]+) steer=([-\d.]+) '
        r'robot=\(([-\d.]+),([-\d.]+)\) person=\(([-\d.]+),([-\d.]+)\) '
        r'spd=([-\d.]+) fps=(\d+) (\d+)ms'
    )

    frames = []
    for line in open(path):
        m = pattern.search(line)
        if m:
            frames.append({
                'time': m.group(1),
                'state': m.group(2),
                'faces': int(m.group(3)),
                'watched': m.group(4) == 'True',
                'bearing': float(m.group(5)),
                'dist': float(m.group(6)),
                'accel': float(m.group(7)),
                'steer': float(m.group(8)),
                'robot_x': float(m.group(9)),
                'robot_y': float(m.group(10)),
                'person_x': float(m.group(11)),
                'person_y': float(m.group(12)),
                'speed': float(m.group(13)),
                'fps': int(m.group(14)),
                'frame_ms': int(m.group(15)),
            })

    # Parse STRIKE events
    strike_pattern = re.compile(r'STRIKE #(\d+) at pos=\(([-\d.]+),([-\d.]+)\)')
    for line in open(path):
        m = strike_pattern.search(line)
        if m:
            frames.append({
                'event': 'STRIKE',
                'kill': int(m.group(1)),
                'x': float(m.group(2)),
                'y': float(m.group(3)),
            })

    return frames


def plot_trajectory(frames: list[dict], grid: np.ndarray | None, output_dir: str):
    """Plot top-down trajectory map."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#111111')

    # Plot occupancy grid if available
    if grid is not None:
        extent_half = grid.shape[0] * 0.1 / 2
        ax.imshow(grid.T, origin='lower', cmap='RdYlGn_r', alpha=0.4,
                  extent=[-extent_half, extent_half, -extent_half, extent_half],
                  vmin=0, vmax=1)

    # Extract positions by state
    data_frames = [f for f in frames if 'state' in f and 'robot_x' in f]
    if not data_frames:
        print("No position data in log")
        return

    states = {'IDLE': '#555555', 'STALKING': '#00ff44', 'FROZEN': '#ff2222', 'STRIKE': '#ffff00'}

    # Plot robot path colored by state
    for i in range(1, len(data_frames)):
        f0 = data_frames[i - 1]
        f1 = data_frames[i]
        color = states.get(f1['state'], '#ffffff')
        ax.plot([f0['robot_x'], f1['robot_x']], [f0['robot_y'], f1['robot_y']],
                color=color, linewidth=1.5, alpha=0.8)

    # Plot person positions when detected
    person_xs = [f['person_x'] for f in data_frames if f['faces'] > 0 and f['person_x'] != 0]
    person_ys = [f['person_y'] for f in data_frames if f['faces'] > 0 and f['person_y'] != 0]
    if person_xs:
        ax.scatter(person_xs, person_ys, c='red', s=15, alpha=0.3, label='Person detected')

    # Plot STRIKE locations
    strikes = [f for f in frames if f.get('event') == 'STRIKE']
    for s in strikes:
        ax.scatter(s['x'], s['y'], c='yellow', s=200, marker='*', zorder=10, edgecolors='red', linewidths=1)

    # Start and end markers
    ax.scatter(data_frames[0]['robot_x'], data_frames[0]['robot_y'],
               c='cyan', s=100, marker='o', zorder=10, label='Start')
    ax.scatter(data_frames[-1]['robot_x'], data_frames[-1]['robot_y'],
               c='magenta', s=100, marker='s', zorder=10, label='End')

    # Legend
    patches = [mpatches.Patch(color=c, label=s) for s, c in states.items()]
    patches.append(mpatches.Patch(color='cyan', label='Start'))
    patches.append(mpatches.Patch(color='yellow', label='Kill'))
    ax.legend(handles=patches, loc='upper right', fontsize=9,
              facecolor='#222222', edgecolor='#444444', labelcolor='white')

    ax.set_title('SCP-173 Hunt Trajectory', color='white', fontsize=16, fontweight='bold')
    ax.set_xlabel('X (meters)', color='white')
    ax.set_ylabel('Y (meters)', color='white')
    ax.tick_params(colors='white')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory.png'), dpi=150, facecolor=fig.get_facecolor())
    print(f"  trajectory.png")
    plt.close()


def plot_state_timeline(frames: list[dict], output_dir: str):
    """Plot state timeline bar chart."""
    data_frames = [f for f in frames if 'state' in f and 'robot_x' in f]
    if not data_frames:
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 3))
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#1a1a1a')

    states = {'IDLE': '#555555', 'STALKING': '#00ff44', 'FROZEN': '#ff2222', 'STRIKE': '#ffff00'}

    for i, f in enumerate(data_frames):
        color = states.get(f['state'], '#ffffff')
        ax.barh(0, 1, left=i, color=color, height=0.8)

    ax.set_yticks([])
    ax.set_xlabel('Frame', color='white')
    ax.set_title('State Timeline', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')

    patches = [mpatches.Patch(color=c, label=s) for s, c in states.items()]
    ax.legend(handles=patches, loc='upper right', fontsize=8, ncol=4,
              facecolor='#222222', edgecolor='#444444', labelcolor='white')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timeline.png'), dpi=150, facecolor=fig.get_facecolor())
    print(f"  timeline.png")
    plt.close()


def plot_speed_profile(frames: list[dict], output_dir: str):
    """Plot speed over time colored by state."""
    data_frames = [f for f in frames if 'state' in f and 'speed' in f]
    if not data_frames:
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    fig.patch.set_facecolor('#111111')
    ax.set_facecolor('#1a1a1a')

    states = {'IDLE': '#555555', 'STALKING': '#00ff44', 'FROZEN': '#ff2222', 'STRIKE': '#ffff00'}

    for i, f in enumerate(data_frames):
        color = states.get(f['state'], '#ffffff')
        ax.bar(i, f['speed'], color=color, width=1.0)

    ax.set_xlabel('Frame', color='white')
    ax.set_ylabel('Speed (m/s)', color='white')
    ax.set_title('Speed Profile', color='white', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speed.png'), dpi=150, facecolor=fig.get_facecolor())
    print(f"  speed.png")
    plt.close()


def compute_stats(frames: list[dict]) -> dict:
    """Compute session statistics."""
    data_frames = [f for f in frames if 'state' in f and 'robot_x' in f]
    if not data_frames:
        return {}

    # Distance traveled
    total_dist = 0
    for i in range(1, len(data_frames)):
        dx = data_frames[i]['robot_x'] - data_frames[i-1]['robot_x']
        dy = data_frames[i]['robot_y'] - data_frames[i-1]['robot_y']
        total_dist += np.sqrt(dx**2 + dy**2)

    # State counts
    state_counts = {}
    for f in data_frames:
        state_counts[f['state']] = state_counts.get(f['state'], 0) + 1
    total_frames = len(data_frames)

    # Kills
    kills = len([f for f in frames if f.get('event') == 'STRIKE'])

    # Stuck events
    stuck = sum(1 for line in open(args.logfile) if 'STUCK' in line)

    # A* paths
    astar_plans = sum(1 for line in open(args.logfile) if 'A* path' in line)

    # FPS
    fps_values = [f['fps'] for f in data_frames if f['fps'] > 0]
    avg_fps = np.mean(fps_values) if fps_values else 0

    # Speed
    speeds = [f['speed'] for f in data_frames]
    peak_speed = max(speeds) if speeds else 0

    # Frame times
    frame_times = [f['frame_ms'] for f in data_frames]

    stats = {
        'total_frames': total_frames,
        'total_distance_m': round(total_dist, 2),
        'kills': kills,
        'stuck_events': stuck,
        'astar_plans': astar_plans,
        'avg_fps': round(avg_fps, 1),
        'peak_speed_ms': round(peak_speed, 2),
        'avg_frame_ms': round(np.mean(frame_times), 0) if frame_times else 0,
        'state_breakdown': {s: f"{100*c/total_frames:.1f}%" for s, c in state_counts.items()},
        'duration_estimate_s': round(total_frames / max(avg_fps, 1), 0),
    }
    return stats


def print_stats(stats: dict, output_dir: str):
    """Print and save stats."""
    lines = [
        "=" * 50,
        "  SCP-173 SESSION REPORT",
        "=" * 50,
        "",
        f"  Duration:        ~{stats.get('duration_estimate_s', 0):.0f} seconds",
        f"  Total frames:    {stats.get('total_frames', 0)}",
        f"  Average FPS:     {stats.get('avg_fps', 0)}",
        f"  Avg frame time:  {stats.get('avg_frame_ms', 0):.0f}ms",
        "",
        f"  Distance:        {stats.get('total_distance_m', 0):.2f} meters",
        f"  Peak speed:      {stats.get('peak_speed_ms', 0):.2f} m/s",
        f"  Kills:           {stats.get('kills', 0)}",
        f"  Stuck events:    {stats.get('stuck_events', 0)}",
        f"  A* paths:        {stats.get('astar_plans', 0)}",
        "",
        "  State breakdown:",
    ]
    for state, pct in stats.get('state_breakdown', {}).items():
        lines.append(f"    {state:12s} {pct}")

    lines.append("")
    lines.append("=" * 50)

    report = "\n".join(lines)
    print(report)

    with open(os.path.join(output_dir, 'report.txt'), 'w') as f:
        f.write(report)
    print(f"  report.txt")


def main():
    global args
    parser = argparse.ArgumentParser(description='SCP-173 Session Analyzer')
    parser.add_argument('logfile', help='Path to scp173.log')
    parser.add_argument('--grid', help='Path to occupancy_grid.npy', default=None)
    parser.add_argument('--output', help='Output directory', default='./scp173_analysis')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Parsing log...")
    frames = parse_log(args.logfile)
    print(f"  {len(frames)} entries")

    grid = None
    if args.grid and os.path.exists(args.grid):
        grid = np.load(args.grid)
        print(f"  Grid loaded: {grid.shape}")

    print("\nGenerating visualizations...")
    plot_trajectory(frames, grid, args.output)
    plot_state_timeline(frames, args.output)
    plot_speed_profile(frames, args.output)

    print("\nSession statistics:")
    stats = compute_stats(frames)
    print_stats(stats, args.output)

    print(f"\nAll outputs saved to {args.output}/")


if __name__ == '__main__':
    main()
