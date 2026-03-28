"""Audio engine — plays SCP-173 sound effects via pygame.mixer.

Sounds:
  scrape_loop.wav  — looping approach sound during STALKING, volume rises as
                     distance decreases
  strike.wav       — one-shot stinger played once on entering STRIKE state

If sound files are missing the engine degrades gracefully (silent mode).
"""

import os
import pygame

from scp173.behavior.state_machine import State
from scp173.config import (
    AUDIO_FREQ, AUDIO_SIZE, AUDIO_CHANNELS, AUDIO_BUFFER,
    STRIKE_SOUND_PATH, SCRAPE_SOUND_PATH,
)


class AudioEngine:
    def __init__(self):
        self._available = False
        try:
            pygame.mixer.init(
                frequency=AUDIO_FREQ,
                size=AUDIO_SIZE,
                channels=AUDIO_CHANNELS,
                buffer=AUDIO_BUFFER,
            )
            self._approach_channel = pygame.mixer.Channel(0)
            self._strike_channel   = pygame.mixer.Channel(1)
            self._scrape  = self._load(SCRAPE_SOUND_PATH)
            self._strike  = self._load(STRIKE_SOUND_PATH)
            self._available = True
        except Exception as e:
            print(f"[AudioEngine] init failed, running silent: {e}")

    # ------------------------------------------------------------------
    def update(self, state: State, distance: float = 1.0) -> None:
        if not self._available:
            return

        if state == State.STALKING:
            if self._scrape and not self._approach_channel.get_busy():
                self._approach_channel.play(self._scrape, loops=-1)
            vol = max(0.05, 1.0 - float(distance))
            self._approach_channel.set_volume(vol)

        elif state == State.STRIKE:
            self._approach_channel.stop()
            if self._strike and not self._strike_channel.get_busy():
                self._strike_channel.play(self._strike)

        else:  # IDLE or FROZEN
            if self._approach_channel.get_busy():
                self._approach_channel.fadeout(500)

    # ------------------------------------------------------------------
    @staticmethod
    def _load(path: str):
        if not os.path.exists(path):
            print(f"[AudioEngine] sound file not found: {path}")
            return None
        try:
            return pygame.mixer.Sound(path)
        except Exception as e:
            print(f"[AudioEngine] could not load {path}: {e}")
            return None
