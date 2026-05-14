"""Modality A (output) — macOS cursor and click events via Quartz CoreGraphics."""

from __future__ import annotations

import logging
from types import ModuleType
from typing import Optional

logger = logging.getLogger(__name__)

_quartz: Optional[ModuleType] = None
try:
    import Quartz as _quartz  # type: ignore[import]
except ImportError:
    logger.warning("Quartz not available; cursor control disabled.")


class CursorController:
    """Moves the macOS cursor and emits click events via CoreGraphics.

    Args:
        screen_width: Display width in pixels.
        screen_height: Display height in pixels.
    """

    def __init__(self, screen_width: int, screen_height: int) -> None:
        self._w = screen_width
        self._h = screen_height

    def move_to(self, x: float, y: float) -> None:
        if _quartz is None:
            return
        x = max(0.0, min(float(x), float(self._w - 1)))
        y = max(0.0, min(float(y), float(self._h - 1)))
        event = _quartz.CGEventCreateMouseEvent(
            None, _quartz.kCGEventMouseMoved, (x, y), _quartz.kCGMouseButtonLeft
        )
        _quartz.CGEventPost(_quartz.kCGHIDEventTap, event)

    def move_normalized(self, nx: float, ny: float) -> None:
        self.move_to(nx * self._w, ny * self._h)

    def click(self) -> None:
        if _quartz is None:
            return
        pos = _quartz.CGEventGetLocation(_quartz.CGEventCreate(None))
        for evt_type in (_quartz.kCGEventLeftMouseDown, _quartz.kCGEventLeftMouseUp):
            evt = _quartz.CGEventCreateMouseEvent(None, evt_type, pos, _quartz.kCGMouseButtonLeft)
            _quartz.CGEventPost(_quartz.kCGHIDEventTap, evt)
