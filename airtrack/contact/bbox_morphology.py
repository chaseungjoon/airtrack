"""Modality A — 2D morphological bounding box for finger-count estimation.

Projects the actively moving fingertips onto the 2D (X, Y) plane and measures
the horizontal span (bounding box width). More fingers produce a statistically
wider horizontal footprint independent of Z-depth.

MediaPipe fingertip landmark indices:
  4  = thumb tip
  8  = index tip
  12 = middle tip
  16 = ring tip
  20 = pinky tip
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from airtrack.shared.hand_tracker import HandLandmarks

FINGERTIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)

# Min horizontal span (normalized) to consider a finger "spread out"
_SPREAD_THRESHOLD = 0.04


class BboxMorphology:
    """Estimates finger count from 2D fingertip bounding box width.

    Args:
        velocity_threshold: Min per-frame velocity for a fingertip to be
            classified as "actively moving" (normalized coords/frame).
    """

    def __init__(self, velocity_threshold: float = 0.005) -> None:
        self._vel_thr = velocity_threshold
        self._prev_tips: Optional[np.ndarray] = None

    def estimate(self, hand: Optional[HandLandmarks]) -> Optional[int]:
        """Return estimated active finger count (1–4) or None if undetermined.

        Args:
            hand: Current frame landmarks.
        """
        if hand is None:
            self._prev_tips = None
            return None

        tips = hand.landmarks[list(FINGERTIP_INDICES), :2]  # (5, 2)

        if self._prev_tips is None:
            self._prev_tips = tips
            return None

        velocities = np.linalg.norm(tips - self._prev_tips, axis=1)  # (5,)
        self._prev_tips = tips

        active = tips[velocities > self._vel_thr]  # rows with significant motion

        if len(active) == 0:
            return None

        bbox_width = float(active[:, 0].max() - active[:, 0].min())
        n_active = len(active)

        # Combine count and spread into a coarse estimate
        if n_active == 1:
            return 1
        if n_active == 2:
            return 1 if bbox_width < _SPREAD_THRESHOLD else 2
        if n_active == 3:
            return 2 if bbox_width < _SPREAD_THRESHOLD * 2 else 3
        # 4 or 5 active → 3 or 4 fingers
        return 3 if bbox_width < _SPREAD_THRESHOLD * 3 else 4

    def reset(self) -> None:
        self._prev_tips = None
