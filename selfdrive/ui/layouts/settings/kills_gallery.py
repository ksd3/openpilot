"""Fullscreen kill photo gallery widget for the comma 4 UI."""

import glob
import os
import time

import pyray as rl
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.widgets import Widget

KILLS_DIR = "/data/openpilot/scp173/kills"


class KillsGalleryWidget(Widget):
  """Fullscreen carousel of kill photos. Tap to advance, swipe down to close."""

  def __init__(self):
    super().__init__()
    self._textures = []
    self._photo_paths = []
    self._idx = 0
    self._last_advance = time.monotonic()
    self._auto_advance = 4.0
    self._font = gui_app.font(FontWeight.BOLD)
    self._loaded = False

  def show_event(self):
    super().show_event()
    self._load_photos()

  def hide_event(self):
    super().hide_event()
    self._unload_photos()

  def _load_photos(self):
    self._photo_paths = sorted(glob.glob(os.path.join(KILLS_DIR, "kill_*.jpg")))
    self._textures = []
    for path in self._photo_paths:
      try:
        img = rl.load_image(path.encode())
        tex = rl.load_texture_from_image(img)
        rl.set_texture_filter(tex, rl.TextureFilter.TEXTURE_FILTER_BILINEAR)
        rl.unload_image(img)
        self._textures.append(tex)
      except Exception:
        pass
    self._idx = 0
    self._last_advance = time.monotonic()
    self._loaded = True

  def _unload_photos(self):
    for tex in self._textures:
      try:
        rl.unload_texture(tex)
      except Exception:
        pass
    self._textures = []
    self._loaded = False

  def _handle_mouse_release(self, mouse_pos):
    if not self._textures:
      gui_app.pop_widget()
      return

    # Tap bottom 20% to close
    screen_h = rl.get_screen_height()
    if mouse_pos.y > screen_h * 0.85:
      gui_app.pop_widget()
      return

    # Tap anywhere else to advance
    self._idx += 1
    self._last_advance = time.monotonic()
    if self._idx >= len(self._textures):
      gui_app.pop_widget()

  def _render(self, rect: rl.Rectangle):
    rl.clear_background(rl.BLACK)

    if not self._textures:
      rl.draw_text_ex(self._font, "No kills yet", rl.Vector2(rect.x + 40, rect.y + rect.height / 2), 60, 0, rl.WHITE)
      return

    # Auto-advance
    if time.monotonic() - self._last_advance > self._auto_advance:
      self._idx += 1
      self._last_advance = time.monotonic()
      if self._idx >= len(self._textures):
        self._idx = 0  # loop

    idx = self._idx % len(self._textures)
    tex = self._textures[idx]

    # Scale to fill screen
    scale_x = rect.width / tex.width
    scale_y = rect.height / tex.height
    scale = max(scale_x, scale_y)
    draw_w = tex.width * scale
    draw_h = tex.height * scale
    draw_x = rect.x + (rect.width - draw_w) / 2
    draw_y = rect.y + (rect.height - draw_h) / 2

    rl.draw_texture_ex(tex, rl.Vector2(draw_x, draw_y), 0, scale, rl.WHITE)

    # Kill counter overlay
    count_text = f"KILL #{idx + 1} / {len(self._textures)}"
    rl.draw_rectangle(int(rect.x), int(rect.y), 400, 60, rl.Color(0, 0, 0, 180))
    rl.draw_text_ex(self._font, count_text, rl.Vector2(rect.x + 15, rect.y + 10), 40, 0, rl.Color(255, 0, 0, 255))

    # Close hint
    hint = "TAP TO ADVANCE  |  TAP BOTTOM TO CLOSE"
    hint_size = 24
    rl.draw_text_ex(self._font, hint, rl.Vector2(rect.x + 10, rect.y + rect.height - 35), hint_size, 0, rl.Color(200, 200, 200, 120))
