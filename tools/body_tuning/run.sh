#!/bin/bash
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
PIDS=()

cleanup() {
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null
}
trap cleanup EXIT

# Start telemetry bridge
python3 "$DIR/body_telemd.py" &
PIDS+=($!)
echo "Started body_telemd.py (PID $!)"

# Start tuning GUI
python3 "$DIR/tune_gui.py" &
PIDS+=($!)
echo "Started tune_gui.py (PID $!)"

# Optionally start PlotJuggler with body tuning layout
if [ "$1" = "--plot" ]; then
  python3 "$DIR/../plotjuggler/juggle.py" --stream --layout "$DIR/static/body_tuning.xml" &
  PIDS+=($!)
  echo "Started PlotJuggler (PID $!)"
fi

echo "Body tuning tools running. Press Ctrl+C to stop."
wait
