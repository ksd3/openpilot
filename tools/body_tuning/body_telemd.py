#!/usr/bin/env python3
"""
Body tuning telemetry daemon.
Subscribes to carControl/carState/carOutput, computes body-specific
derived signals, and publishes them on the bodyTuningTelemetry cereal
service for PlotJuggler to consume via its Cereal Subscriber plugin.
"""
import time

import cereal.messaging as messaging

SPEED_FROM_RPM = 0.008587


def main():
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
