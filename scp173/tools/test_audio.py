#!/usr/bin/env python3
"""Test audio playback on the comma 4."""
import subprocess
import sys
import os

SOUNDS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sounds")

def play(path):
    print(f"Playing {os.path.basename(path)}...")
    # Try aplay first (ALSA)
    result = subprocess.run(["aplay", path], capture_output=True, timeout=10)
    if result.returncode == 0:
        print("  aplay: OK")
        return True
    print(f"  aplay failed: {result.stderr.decode().strip()}")

    # Try paplay (PulseAudio)
    result = subprocess.run(["paplay", path], capture_output=True, timeout=10)
    if result.returncode == 0:
        print("  paplay: OK")
        return True
    print(f"  paplay failed: {result.stderr.decode().strip()}")

    # Try ffplay
    result = subprocess.run(["ffplay", "-nodisp", "-autoexit", path], capture_output=True, timeout=10)
    if result.returncode == 0:
        print("  ffplay: OK")
        return True
    print(f"  ffplay failed: {result.stderr.decode().strip()}")

    return False

if __name__ == "__main__":
    # Check what audio tools exist
    for tool in ["aplay", "paplay", "ffplay", "speaker-test"]:
        r = subprocess.run(["which", tool], capture_output=True)
        print(f"{tool}: {'found' if r.returncode == 0 else 'not found'}")

    print()
    for wav in sorted(os.listdir(SOUNDS_DIR)):
        if wav.endswith(".wav"):
            play(os.path.join(SOUNDS_DIR, wav))
