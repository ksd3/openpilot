#!/usr/bin/env python3
"""
Send a forward drive command to comma body v2 via testJoystick.

Usage:
  # On the comma v4 device (SSH in):
  cereal/messaging/bridge <your_pc_ip> testJoystick

  # On your PC:
  python tools/camerastream/move_forward.py 192.168.63.120
  python tools/camerastream/move_forward.py 192.168.63.120 --throttle -0.5 --duration 3
"""

import argparse
import time

import zmq

from cereal import log


def get_port(endpoint: str) -> int:
  fnv_prime = 0x100000001b3
  hash_value = 0xcbf29ce484222325
  for c in endpoint.encode():
    hash_value ^= c
    hash_value = (hash_value * fnv_prime) & 0xFFFFFFFFFFFFFFFF
  return 8023 + (hash_value % (65535 - 8023))


def build_joystick_msg(accel: float, steer: float) -> bytes:
  msg = log.Event.new_message()
  msg.logMonoTime = int(time.monotonic() * 1e9)
  msg.valid = True
  msg.init('testJoystick')
  msg.testJoystick.axes = [float(accel), float(steer)]
  msg.testJoystick.buttons = []
  return msg.to_bytes()


def main():
  parser = argparse.ArgumentParser(description="Drive comma body forward")
  parser.add_argument("addr", help="Comma v4 IP address")
  parser.add_argument("--throttle", type=float, default=-0.3,
                      help="Throttle value (negative = forward, default: -0.3)")
  parser.add_argument("--steer", type=float, default=0.0,
                      help="Steering value (-1 to 1, default: 0.0)")
  parser.add_argument("--duration", type=float, default=2.0,
                      help="How long to drive in seconds (default: 2.0)")
  args = parser.parse_args()

  joystick_port = get_port("testJoystick")
  ctx = zmq.Context()
  pub_sock = ctx.socket(zmq.PUB)
  pub_sock.bind(f"tcp://0.0.0.0:{joystick_port}")

  print(f"Publishing testJoystick on port {joystick_port}")
  print(f"Throttle: {args.throttle}, Steer: {args.steer}, Duration: {args.duration}s")
  print("Waiting 1s for subscriber to connect...")
  time.sleep(1.0)

  print("Driving...")
  start = time.monotonic()
  while time.monotonic() - start < args.duration:
    msg = build_joystick_msg(args.throttle, args.steer)
    pub_sock.send(msg)
    time.sleep(0.01)  # 100Hz

  # send stop
  print("Stopping...")
  for _ in range(50):
    pub_sock.send(build_joystick_msg(0.0, 0.0))
    time.sleep(0.01)

  print("Done.")
  pub_sock.close()
  ctx.term()


if __name__ == "__main__":
  main()
