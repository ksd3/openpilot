"""Motor controller — publishes testJoystick messages consumed by joystickd.

joystickd is already running as part of openpilot on the body and publishes
carControl. We feed it via testJoystick to avoid publisher conflicts.

testJoystick.axes = [accel, steer]
  accel: -1.0 = full forward,  +1.0 = full backward  (joystickd convention)
  steer: -1.0 = full left,     +1.0 = full right
"""

import numpy as np
from cereal import messaging


class MotorController:
    def __init__(self):
        self._pm = messaging.PubMaster(["testJoystick"])

    def send(self, accel: float, steer: float) -> None:
        # joystickd axes[0]: -1 = forward, so negate our accel
        joystick_accel = float(np.clip(-accel, -1.0, 1.0))
        joystick_steer = float(np.clip(steer,  -1.0, 1.0))

        msg = messaging.new_message("testJoystick")
        msg.valid = True
        msg.testJoystick.axes = [joystick_accel, joystick_steer]
        self._pm.send("testJoystick", msg)

    def stop(self) -> None:
        self.send(0.0, 0.0)
