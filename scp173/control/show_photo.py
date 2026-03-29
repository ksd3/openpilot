#!/usr/bin/env python3
"""Display a kill photo on the comma 4 screen for a few seconds.

Kills the UI, takes over the screen with pyray, shows the photo,
then exits so the UI can restart.

Usage: python show_photo.py <image_path> [duration_seconds]
"""

import subprocess
import sys
import time


def show_photo(image_path: str, duration: float = 3.0):
    # Kill UI to free the screen
    subprocess.run(["pkill", "-f", "selfdrive.ui.ui"], capture_output=True)
    time.sleep(0.3)

    try:
        import pyray as rl

        rl.init_window(0, 0, "SCP-173 KILL")
        screen_w = rl.get_screen_width()
        screen_h = rl.get_screen_height()

        img = rl.load_image(image_path.encode())
        # Scale to fit screen
        rl.image_resize(img, screen_w, screen_h)
        tex = rl.load_texture_from_image(img)
        rl.unload_image(img)

        start = time.monotonic()
        while time.monotonic() - start < duration:
            rl.begin_drawing()
            rl.clear_background(rl.BLACK)
            rl.draw_texture(tex, 0, 0, rl.WHITE)

            # "KILLED" text overlay
            font_size = 80
            text = "ELIMINATED"
            text_w = rl.measure_text(text.encode(), font_size)
            rl.draw_text(text.encode(), (screen_w - text_w) // 2, screen_h - 120,
                        font_size, rl.Color(255, 0, 0, 255))
            rl.end_drawing()

        rl.unload_texture(tex)
        rl.close_window()
    except Exception as e:
        print(f"show_photo error: {e}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    if path:
        show_photo(path, dur)
