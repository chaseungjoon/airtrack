"""Modality C — Pinch-to-click detection via thumb-index Euclidean distance.

When the Euclidean distance between MediaPipe Landmark 4 (thumb tip) and
Landmark 8 (index tip) falls below an empirical threshold, a click event
is triggered and held until distance exceeds the release threshold.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from airtrack.shared.hand_tracker import HandLandmarks

THUMB_TIP: int = 4
INDEX_TIP: int = 8

_DEFAULT_CLICK_THRESH = 0.04    # normalized distance to trigger click
_DEFAULT_RELEASE_THRESH = 0.06  # normalized distance to release click


class PinchDetector:
    """Detects thumb-index pinch gestures and emits click state transitions.

    Args:
        click_threshold: Distance (normalized) below which a click is registered.
        release_threshold: Distance above which click is released (hysteresis).
    """

    def __init__(
        self,
        click_threshold: float = _DEFAULT_CLICK_THRESH,
        release_threshold: float = _DEFAULT_RELEASE_THRESH,
    ) -> None:
        self._click_thr = click_threshold
        self._release_thr = release_threshold
        self._is_clicking = False

    @property
    def is_clicking(self) -> bool:
        return self._is_clicking

    def update(self, hand: Optional[HandLandmarks]) -> bool:
        """Process one frame and return click state.

        Args:
            hand: Latest hand landmarks, or None if no hand visible.

        Returns:
            True if a click is currently active.
        """
        if hand is None:
            self._is_clicking = False
            return False

        thumb = hand.landmarks[THUMB_TIP, :2]
        index = hand.landmarks[INDEX_TIP, :2]
        dist = float(np.linalg.norm(thumb - index))

        if not self._is_clicking and dist < self._click_thr:
            self._is_clicking = True
        elif self._is_clicking and dist > self._release_thr:
            self._is_clicking = False

        return self._is_clicking
