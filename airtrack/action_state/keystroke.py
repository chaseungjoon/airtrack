"""Modality A — OS-level keystroke rate monitor via pynput.

Provides the primary enabling gate for Module 1: if keystroke_rate > threshold,
the user is typing and gesture pipeline is suppressed.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

from pynput import keyboard

from airtrack.config import WINDOW_DURATION_SEC


class KeystrokeMonitor:
    """Listens for key-press events and computes a rolling keystroke rate.

    Runs on a background thread; safe to call from the main video loop.

    Args:
        window_sec: Rolling window duration in seconds.
    """

    def __init__(self, window_sec: float = WINDOW_DURATION_SEC) -> None:
        self._window_sec = window_sec
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()
        self._listener: Optional[keyboard.Listener] = None

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        now = time.monotonic()
        with self._lock:
            self._timestamps.append(now)
            cutoff = now - self._window_sec
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

    @property
    def keystroke_rate(self) -> float:
        """Keystrokes per second over the recent rolling window."""
        now = time.monotonic()
        with self._lock:
            cutoff = now - self._window_sec
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()
            count = len(self._timestamps)
        return count / self._window_sec if self._window_sec > 0 else 0.0

    def start(self) -> None:
        self._listener = keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def __enter__(self) -> "KeystrokeMonitor":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
