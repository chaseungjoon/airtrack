"""macOS cursor movement via Quartz CoreGraphics event API.

Requires pyobjc-framework-Quartz.
"""

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
    """Moves the macOS mouse cursor using CoreGraphics synthetic events.

    Coordinates are in screen pixels with origin at the top-left corner.

    Args:
        screen_width: Display width in pixels (used for bounds clamping).
        screen_height: Display height in pixels.
    """

    def __init__(self, screen_width: int, screen_height: int) -> None:
        self._screen_width = screen_width
        self._screen_height = screen_height

    def move_to(self, x: float, y: float) -> None:
        """Move the cursor to absolute screen coordinates.

        Args:
            x: Horizontal screen position in pixels.
            y: Vertical screen position in pixels.
        """
        if _quartz is None:
            return
        x = max(0.0, min(float(x), float(self._screen_width - 1)))
        y = max(0.0, min(float(y), float(self._screen_height - 1)))
        event = _quartz.CGEventCreateMouseEvent(
            None,
            _quartz.kCGEventMouseMoved,
            (x, y),
            _quartz.kCGMouseButtonLeft,
        )
        _quartz.CGEventPost(_quartz.kCGHIDEventTap, event)

    def move_normalized(self, nx: float, ny: float) -> None:
        """Move the cursor using normalized [0, 1] coordinates.

        Args:
            nx: Horizontal position in [0, 1].
            ny: Vertical position in [0, 1].
        """
        self.move_to(nx * self._screen_width, ny * self._screen_height)

    def click(self) -> None:
        """Emit a left mouse click at the current cursor position."""
        if _quartz is None:
            return
        pos = _quartz.CGEventGetLocation(_quartz.CGEventCreate(None))
        for event_type in (
            _quartz.kCGEventLeftMouseDown,
            _quartz.kCGEventLeftMouseUp,
        ):
            evt = _quartz.CGEventCreateMouseEvent(
                None, event_type, pos, _quartz.kCGMouseButtonLeft
            )
            _quartz.CGEventPost(_quartz.kCGHIDEventTap, evt)
