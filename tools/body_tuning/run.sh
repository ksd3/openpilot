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

# Start telemetry bridge + param receiver (runs on device)
python3 "$DIR/body_telemd.py" &
PIDS+=($!)
echo "Started body_telemd.py (PID $!)"

DEVICE_IP=$(hostname -I | awk '{print $1}')
echo ""
echo "Body tuning daemon running. Press Ctrl+C to stop."
echo ""
echo "On your PC, run:"
echo "  python tune_gui.py $DEVICE_IP"
echo "  ZMQ=1 ./tools/plotjuggler/juggle.py --stream --layout tools/body_tuning/static/body_tuning.xml"
echo ""
echo "In PlotJuggler, select Cereal Subscriber and enter device IP: $DEVICE_IP"
wait
