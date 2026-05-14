"""Module 2 fusion: combines top-down vision, motion parsing, and pinch detection
to emit (dx, dy, click_state) HID commands.

Activated only when Module 1 sets is_gesture_active = True.
"""

from __future__ import annotations

from typing import Optional

from airtrack.shared.hand_tracker import HandLandmarks
from airtrack.kinematic.motion_parser import MotionParser
from airtrack.kinematic.pinch import PinchDetector
from airtrack.kinematic.hid_bridge import CursorController


class KinematicDecoder:
    """Translates hand landmarks into cursor movement and click events.

    Args:
        controller: CursorController for OS event injection.
        speed_scale: Multiplier on normalized velocity for cursor sensitivity.
    """

    def __init__(
        self,
        controller: Optional[CursorController] = None,
        speed_scale: float = 3.0,
    ) -> None:
        self._parser = MotionParser(speed_scale=speed_scale)
        self._pinch = PinchDetector()
        self._controller = controller
        self._cursor_x = 0.5
        self._cursor_y = 0.5

    def update(self, hand: Optional[HandLandmarks]) -> tuple[float, float, bool]:
        """Process one frame; optionally inject OS events.

        Args:
            hand: Latest hand landmarks from the primary (right) hand.

        Returns:
            (dx, dy, click_state) where dx/dy are normalized velocity deltas.
        """
        dx, dy = self._parser.update(hand)
        click = self._pinch.update(hand)

        if self._controller is not None and hand is not None:
            self._cursor_x = max(0.0, min(1.0, self._cursor_x + dx))
            self._cursor_y = max(0.0, min(1.0, self._cursor_y + dy))
            self._controller.move_normalized(self._cursor_x, self._cursor_y)
            if click:
                self._controller.click()

        return dx, dy, click

    def reset(self) -> None:
        self._parser.reset()
        self._cursor_x = 0.5
        self._cursor_y = 0.5
