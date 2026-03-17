#!/usr/bin/env python3
"""
Tkinter GUI for live-tuning body PID gains via openpilot Params.
"""
import tkinter as tk

from openpilot.common.params import Params

GAINS = {
  'BodySpeedPidKp': {'label': 'Speed Kp', 'default': 110.0, 'max': 330.0, 'resolution': 0.5},
  'BodySpeedPidKi': {'label': 'Speed Ki', 'default': 11.5, 'max': 34.5, 'resolution': 0.1},
  'BodyTurnPidKp':  {'label': 'Turn Kp',  'default': 150.0, 'max': 450.0, 'resolution': 0.5},
  'BodyTurnPidKi':  {'label': 'Turn Ki',  'default': 15.0,  'max': 45.0,  'resolution': 0.1},
}


class TuningGUI:
  def __init__(self):
    self.params = Params()
    self.root = tk.Tk()
    self.root.title("Body PID Tuning")
    self.sliders: dict[str, tk.Scale] = {}
    self.labels: dict[str, tk.Label] = {}
    self._write_pending = False

    for i, (key, cfg) in enumerate(GAINS.items()):
      initial = self._read_param(key, cfg['default'])

      tk.Label(self.root, text=cfg['label'], font=('monospace', 12), width=12, anchor='w').grid(row=i, column=0, padx=(10, 5), pady=5)

      s = tk.Scale(self.root, from_=0, to=cfg['max'], resolution=cfg['resolution'],
                   orient=tk.HORIZONTAL, length=400, showvalue=False,
                   command=lambda v, k=key: self._on_change(k, float(v)))
      s.set(initial)
      s.grid(row=i, column=1, padx=5, pady=5)
      self.sliders[key] = s

      lbl = tk.Label(self.root, text=f"{initial:.1f}", font=('monospace', 12), width=8)
      lbl.grid(row=i, column=2, padx=(5, 10), pady=5)
      self.labels[key] = lbl

    btn_frame = tk.Frame(self.root)
    btn_frame.grid(row=len(GAINS), column=0, columnspan=3, pady=10)
    tk.Button(btn_frame, text="Reset to Defaults", command=self._reset).pack()

    self._flush_params()
    self.root.mainloop()

  def _read_param(self, key: str, default: float) -> float:
    try:
      val = self.params.get(key, return_default=True)
      if val is not None:
        return float(val)
    except Exception:
      pass
    return default

  def _on_change(self, key: str, value: float):
    self.labels[key].config(text=f"{value:.1f}")
    if not self._write_pending:
      self._write_pending = True
      self.root.after(100, self._flush_params)

  def _flush_params(self):
    self._write_pending = False
    for key in GAINS:
      val = self.sliders[key].get()
      self.params.put_nonblocking(key, str(val))

  def _reset(self):
    for key, cfg in GAINS.items():
      self.sliders[key].set(cfg['default'])
      self.labels[key].config(text=f"{cfg['default']:.1f}")
    self._flush_params()


if __name__ == '__main__':
  TuningGUI()
