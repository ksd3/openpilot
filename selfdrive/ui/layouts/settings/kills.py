"""SCP-173 Kills Gallery — settings panel showing trophy photos."""

import glob
import os
import subprocess

import pyray as rl
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.button import Button, ButtonStyle
from openpilot.system.ui.widgets.list_view import text_item, button_item
from openpilot.system.ui.widgets.scroller_tici import Scroller

KILLS_DIR = "/data/openpilot/scp173/kills"
GALLERY_SCRIPT = "/data/openpilot/scp173/control/show_photo.py"


class KillsLayout(Widget):
  def __init__(self):
    super().__init__()
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._items = []
    self._scroller = None
    self._build_list()

  def _get_kills(self) -> list[str]:
    return sorted(glob.glob(os.path.join(KILLS_DIR, "kill_*.jpg")))

  def _build_list(self):
    kills = self._get_kills()
    items = []

    # Header with count
    items.append(text_item(
      lambda k=len(kills): f"Total Kills: {k}",
      lambda k=len(kills): f"{k} targets eliminated by SCP-173"
    ))

    # Gallery button
    if kills:
      items.append(button_item(
        lambda: "View Kill Gallery",
        lambda: "GALLERY",
        lambda k=len(kills): f"Browse all {k} kill photos fullscreen",
        callback=self._launch_gallery
      ))

    # Individual kills
    for i, path in enumerate(reversed(kills)):
      fname = os.path.basename(path)
      # Extract timestamp from filename: kill_001_1774760324.jpg
      parts = fname.replace(".jpg", "").split("_")
      kill_num = int(parts[1]) if len(parts) > 1 else i + 1

      items.append(button_item(
        lambda n=kill_num: f"Kill #{n}",
        lambda: "VIEW",
        lambda p=path: f"{os.path.basename(p)}",
        callback=lambda p=path: self._view_single(p)
      ))

    self._items = items
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _launch_gallery(self):
    subprocess.Popen([
      "/usr/local/venv/bin/python", GALLERY_SCRIPT, "--gallery", "4"
    ], env={**os.environ, "PYTHONPATH": "/data/openpilot"})

  def _view_single(self, path: str):
    subprocess.Popen([
      "/usr/local/venv/bin/python", GALLERY_SCRIPT, path, "5"
    ], env={**os.environ, "PYTHONPATH": "/data/openpilot"})

  def _render(self, rect: rl.Rectangle):
    # Refresh list each render in case new kills appeared
    current_count = len(self._get_kills())
    if len(self._items) != current_count + 2:  # header + gallery button + kills
      self._build_list()

    if self._scroller:
      self._scroller.render(rect)
