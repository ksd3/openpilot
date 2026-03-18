#!/usr/bin/env python3
"""
Tkinter GUI for live-tuning body PID gains (runs on remote computer).
Sends param updates over ZMQ to body_telemd.py on the device.

Usage: python tune_gui.py <device_ip>
"""
import argparse
import json
import tkinter as tk

import zmq

PARAM_SEND_PORT = 8290

GAINS = {
  'BodySpeedPidKp': {'label': 'Speed Kp', 'default': 110.0, 'max': 330.0, 'resolution': 0.5},
  'BodySpeedPidKi': {'label': 'Speed Ki', 'default': 11.5, 'max': 34.5, 'resolution': 0.1},
  'BodyTurnPidKp':  {'label': 'Turn Kp',  'default': 150.0, 'max': 450.0, 'resolution': 0.5},
  'BodyTurnPidKi':  {'label': 'Turn Ki',  'default': 15.0,  'max': 45.0,  'resolution': 0.1},
}


class TuningGUI:
  def __init__(self, device_ip: str):
    self.ctx = zmq.Context()
    self.sock = self.ctx.socket(zmq.PUSH)
    self.sock.connect(f"tcp://{device_ip}:{PARAM_SEND_PORT}")

    self.root = tk.Tk()
    self.root.title(f"Body PID Tuning — {device_ip}")
    self.sliders: dict[str, tk.Scale] = {}
    self.labels: dict[str, tk.Label] = {}
    self._write_pending = False

    for i, (key, cfg) in enumerate(GAINS.items()):
      tk.Label(self.root, text=cfg['label'], font=('monospace', 12), width=12, anchor='w').grid(row=i, column=0, padx=(10, 5), pady=5)

      s = tk.Scale(self.root, from_=0, to=cfg['max'], resolution=cfg['resolution'],
                   orient=tk.HORIZONTAL, length=400, showvalue=False,
                   command=lambda v, k=key: self._on_change(k, float(v)))
      s.set(cfg['default'])
      s.grid(row=i, column=1, padx=5, pady=5)
      self.sliders[key] = s

      lbl = tk.Label(self.root, text=f"{cfg['default']:.1f}", font=('monospace', 12), width=8)
      lbl.grid(row=i, column=2, padx=(5, 10), pady=5)
      self.labels[key] = lbl

    btn_frame = tk.Frame(self.root)
    btn_frame.grid(row=len(GAINS), column=0, columnspan=3, pady=10)
    tk.Button(btn_frame, text="Reset to Defaults", command=self._reset).pack()

    self._flush_params()
    self.root.mainloop()

  def _on_change(self, key: str, value: float):
    self.labels[key].config(text=f"{value:.1f}")
    if not self._write_pending:
      self._write_pending = True
      self.root.after(100, self._flush_params)

  def _flush_params(self):
    self._write_pending = False
    data = {key: self.sliders[key].get() for key in GAINS}
    self.sock.send_string(json.dumps(data))

  def _reset(self):
    for key, cfg in GAINS.items():
      self.sliders[key].set(cfg['default'])
      self.labels[key].config(text=f"{cfg['default']:.1f}")
    self._flush_params()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Body PID tuning GUI')
  parser.add_argument('device_ip', help='IP address of the device running body_telemd.py')
  args = parser.parse_args()
  TuningGUI(args.device_ip)
