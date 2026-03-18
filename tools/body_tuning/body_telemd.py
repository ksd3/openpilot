#!/usr/bin/env python3
"""
Body tuning telemetry daemon (runs on device).

1. Subscribes to carControl/carState/carOutput, computes body-specific
   derived signals, and publishes them on the bodyTuningTelemetry cereal
   service for PlotJuggler to consume via its Cereal Subscriber plugin.

2. Listens on a ZMQ PULL socket (port 8290) for PID param updates from
   the remote tune_gui.py, and writes them to openpilot Params.
"""
import json
import threading

import zmq

import cereal.messaging as messaging
from openpilot.common.params import Params

SPEED_FROM_RPM = 0.008587
PARAM_RECV_PORT = 8290

VALID_PARAMS = {'BodySpeedPidKp', 'BodySpeedPidKi', 'BodyTurnPidKp', 'BodyTurnPidKi'}


def param_receiver(params: Params):
  """Thread: receives PID param updates over ZMQ and writes to Params."""
  ctx = zmq.Context()
  sock = ctx.socket(zmq.PULL)
  sock.bind(f"tcp://*:{PARAM_RECV_PORT}")
  print(f"Listening for param updates on tcp://*:{PARAM_RECV_PORT}")

  while True:
    try:
      data = json.loads(sock.recv_string())
      for key, val in data.items():
        if key in VALID_PARAMS:
          params.put_nonblocking(key, str(float(val)))
          print(f"  {key} = {val}")
    except Exception as e:
      print(f"Param receive error: {e}")


def main():
  params = Params()

  # Start param receiver thread
  t = threading.Thread(target=param_receiver, args=(params,), daemon=True)
  t.start()

  # Telemetry bridge
  sm = messaging.SubMaster(['carControl', 'carState', 'carOutput'])
  pm = messaging.PubMaster(['bodyTuningTelemetry'])

  while True:
    sm.update()

    if sm.updated['carControl'] or sm.updated['carState']:
      cc = sm['carControl']
      cs = sm['carState']
      co = sm['carOutput']

      speed_desired = cc.actuators.accel / 5.0
      speed_measured = SPEED_FROM_RPM * (cs.wheelSpeeds.fl + cs.wheelSpeeds.fr) / 2.0
      speed_error = speed_desired - speed_measured

      speed_diff_desired = -cc.actuators.torque / 1.5
      speed_diff_measured = SPEED_FROM_RPM * (cs.wheelSpeeds.fl - cs.wheelSpeeds.fr)

      torque_l = co.actuatorsOutput.accel
      torque_r = co.actuatorsOutput.torque

      msg = messaging.new_message('bodyTuningTelemetry')
      msg.bodyTuningTelemetry.speedDesired = speed_desired
      msg.bodyTuningTelemetry.speedMeasured = speed_measured
      msg.bodyTuningTelemetry.speedError = speed_error
      msg.bodyTuningTelemetry.speedDiffDesired = speed_diff_desired
      msg.bodyTuningTelemetry.speedDiffMeasured = speed_diff_measured
      msg.bodyTuningTelemetry.torqueL = torque_l
      msg.bodyTuningTelemetry.torqueR = torque_r
      pm.send('bodyTuningTelemetry', msg)


if __name__ == '__main__':
  main()
