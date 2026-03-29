#!/usr/bin/env python3
"""Display kill photos on the comma 4 screen.

Single photo:  python show_photo.py <path> [seconds]
Gallery:       python show_photo.py --gallery [seconds_per_photo]

In gallery mode, cycles through all photos in scp173/kills/.
Tap screen to advance, or wait for auto-advance.
"""

import glob
import os
import subprocess
import sys
import time

KILLS_DIR = "/data/openpilot/scp173/kills"


def show_gallery(seconds_per: float = 4.0):
    photos = sorted(glob.glob(os.path.join(KILLS_DIR, "kill_*.jpg")))
    if not photos:
        return

    # Kill UI to free screen
    subprocess.run(["pkill", "-f", "selfdrive.ui.ui"], capture_output=True)
    time.sleep(0.3)

    try:
        import pyray as rl

        rl.init_window(0, 0, "SCP-173 KILLS")
        screen_w = rl.get_screen_width()
        screen_h = rl.get_screen_height()

        # Load all textures
        textures = []
        for path in photos:
            img = rl.load_image(path.encode())
            rl.image_resize(img, screen_w, screen_h)
            tex = rl.load_texture_from_image(img)
            rl.unload_image(img)
            textures.append(tex)

        idx = 0
        last_advance = time.monotonic()
        font_size = 60
        count_font_size = 40

        while idx < len(textures):
            if rl.window_should_close():
                break

            rl.begin_drawing()
            rl.clear_background(rl.BLACK)
            rl.draw_texture(textures[idx], 0, 0, rl.WHITE)

            # Kill number overlay
            kill_text = f"KILL #{idx + 1} / {len(textures)}"
            text_w = rl.measure_text(kill_text.encode(), count_font_size)
            rl.draw_rectangle(screen_w - text_w - 30, 10, text_w + 20, count_font_size + 10,
                            rl.Color(0, 0, 0, 180))
            rl.draw_text(kill_text.encode(), screen_w - text_w - 20, 15,
                        count_font_size, rl.Color(255, 0, 0, 255))

            # "TAP FOR NEXT" hint
            if len(textures) > 1:
                hint = "TAP FOR NEXT"
                hint_w = rl.measure_text(hint.encode(), 30)
                rl.draw_text(hint.encode(), (screen_w - hint_w) // 2, screen_h - 50,
                            30, rl.Color(200, 200, 200, 150))

            rl.end_drawing()

            # Advance on tap or timer
            if rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT):
                idx += 1
                last_advance = time.monotonic()
            elif time.monotonic() - last_advance > seconds_per:
                idx += 1
                last_advance = time.monotonic()

        for tex in textures:
            rl.unload_texture(tex)
        rl.close_window()

    except Exception as e:
        print(f"gallery error: {e}")


def show_single(image_path: str, duration: float = 3.0):
    subprocess.run(["pkill", "-f", "selfdrive.ui.ui"], capture_output=True)
    time.sleep(0.3)

    try:
        import pyray as rl

        rl.init_window(0, 0, "SCP-173 KILL")
        screen_w = rl.get_screen_width()
        screen_h = rl.get_screen_height()

        img = rl.load_image(image_path.encode())
        rl.image_resize(img, screen_w, screen_h)
        tex = rl.load_texture_from_image(img)
        rl.unload_image(img)

        start = time.monotonic()
        while time.monotonic() - start < duration:
            rl.begin_drawing()
            rl.clear_background(rl.BLACK)
            rl.draw_texture(tex, 0, 0, rl.WHITE)

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
    if "--gallery" in sys.argv:
        dur = float(sys.argv[sys.argv.index("--gallery") + 1]) if len(sys.argv) > sys.argv.index("--gallery") + 1 else 4.0
        show_gallery(dur)
    elif len(sys.argv) > 1:
        path = sys.argv[1]
        dur = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
        show_single(path, dur)
