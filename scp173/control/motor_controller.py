"""Motor controller — publishes carControl directly, bypassing joystickd.

Publishes carControl with enabled=True, longActive=True, latActive=True
so the body carcontroller always receives commands regardless of
selfdriveState engagement status.
"""

import numpy as np
from cereal import messaging


class MotorController:
    def __init__(self):
        self._pm = messaging.PubMaster(["carControl"])

    def send(self, accel: float, steer: float) -> None:
        # Body v2 carcontroller expects:
        #   actuators.accel: divided by 4, multiplied by MAX_VELOCITY → speed target
        #   actuators.torque: multiplied by MAX_TURN_RATE → turn rate target
        # Negative accel = forward for the body
        joystick_accel = float(np.clip(-accel, -1.0, 1.0))
        joystick_steer = float(np.clip(steer, -1.0, 1.0))

        msg = messaging.new_message("carControl")
        msg.valid = True
        cc = msg.carControl
        cc.enabled = True
        cc.longActive = True
        cc.latActive = True
        cc.actuators.accel = 4.0 * joystick_accel
        cc.actuators.torque = joystick_steer
        self._pm.send("carControl", msg)

    def stop(self) -> None:
        self.send(0.0, 0.0)
