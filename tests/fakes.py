"""Reusable in-process test doubles for AirTrack subsystems.

These avoid any real hardware, OS events, or PyObjC calls in tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Optional

import numpy as np

from airtrack.shared.hand_tracker import HandLandmarks


# ---------------------------------------------------------------------------
# Hand landmark helpers
# ---------------------------------------------------------------------------

def make_hand(
    index_x: float = 0.5,
    index_y: float = 0.5,
    index_z: float = 0.0,
    handedness: str = "Right",
) -> HandLandmarks:
    """Return a HandLandmarks with only the index fingertip positioned.

    All other landmarks are placed at the origin.

    Args:
        index_x: Normalised x of the index fingertip (landmark 8).
        index_y: Normalised y.
        index_z: Normalised z depth.
        handedness: "Left" or "Right".
    """
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[8] = [index_x, index_y, index_z]
    lm[4] = [index_x - 0.05, index_y - 0.05, 0.0]  # thumb tip
    return HandLandmarks(landmarks=lm, handedness=handedness)


def make_hand_sequence(
    positions: list[tuple[float, float]],
    handedness: str = "Right",
) -> list[HandLandmarks]:
    """Create a sequence of HandLandmarks from a list of (x, y) positions.

    Args:
        positions: List of (index_x, index_y) pairs.
        handedness: Applied to every frame.
    """
    return [make_hand(x, y, handedness=handedness) for x, y in positions]


def blank_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Return an all-black BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Fake camera
# ---------------------------------------------------------------------------

class FakeCamera:
    """Yields pre-supplied BGR frames instead of reading from a device.

    Args:
        frames: Ordered list of BGR frames to serve. Loops forever if
            ``loop`` is True, otherwise stops after the last frame.
        loop: Whether to cycle through frames repeatedly.
    """

    def __init__(
        self,
        frames: Optional[list[np.ndarray]] = None,
        loop: bool = True,
    ) -> None:
        self._frames = frames or [blank_frame()]
        self._loop = loop
        self._pos = 0

    def read(self) -> Optional[np.ndarray]:
        if self._pos >= len(self._frames):
            if self._loop:
                self._pos = 0
            else:
                return None
        frame = self._frames[self._pos]
        self._pos += 1
        return frame

    def stream(self) -> Iterator[np.ndarray]:
        while True:
            f = self.read()
            if f is None:
                return
            yield f

    def open(self) -> None:
        pass

    def release(self) -> None:
        pass

    def __enter__(self) -> "FakeCamera":
        return self

    def __exit__(self, *_: object) -> None:
        pass


# ---------------------------------------------------------------------------
# Fake hand tracker
# ---------------------------------------------------------------------------

class FakeHandTracker:
    """Returns a pre-programmed sequence of hand landmark results.

    Args:
        results: List of per-frame results (each a list of HandLandmarks).
            When exhausted, returns empty list.
    """

    def __init__(self, results: Optional[list[list[HandLandmarks]]] = None) -> None:
        self._results = results or []
        self._pos = 0

    def process(self, frame: np.ndarray) -> list[HandLandmarks]:  # noqa: ARG002
        if self._pos >= len(self._results):
            return []
        result = self._results[self._pos]
        self._pos += 1
        return result

    def close(self) -> None:
        pass

    def __enter__(self) -> "FakeHandTracker":
        return self

    def __exit__(self, *_: object) -> None:
        pass


# ---------------------------------------------------------------------------
# Fake haptic feedback (records calls for assertion)
# ---------------------------------------------------------------------------

class FakeHapticFeedback:
    """Captures haptic trigger calls instead of hitting the Taptic Engine."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def trigger(self, pattern: str = "generic") -> None:
        self.calls.append(pattern)


# ---------------------------------------------------------------------------
# Fake cursor controller (records calls for assertion)
# ---------------------------------------------------------------------------

class FakeCursorController:
    """Captures cursor move calls instead of posting Quartz events."""

    def __init__(self) -> None:
        self.positions: list[tuple[float, float]] = []

    def move_to(self, x: float, y: float) -> None:
        self.positions.append((x, y))

    def move_normalized(self, nx: float, ny: float) -> None:
        self.positions.append((nx, ny))

    def click(self) -> None:
        pass
