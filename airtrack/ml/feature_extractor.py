"""Extracts per-frame velocity and depth features from hand landmark sequences."""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from airtrack.vision.hand_tracker import HandLandmarks
from airtrack.config import WINDOW_DURATION_SEC, TARGET_FPS

# Index-finger fingertip landmark index in MediaPipe 21-point model
INDEX_FINGERTIP: int = 8
# Thumb tip
THUMB_TIP: int = 4

_WINDOW_FRAMES: int = max(1, int(WINDOW_DURATION_SEC * TARGET_FPS))


class FeatureVector(np.ndarray):
    """Alias for a 1-D float32 feature array passed to the classifier."""


class FeatureExtractor:
    """Computes a fixed-length feature vector from a rolling landmark window.

    Features per frame (7 values):
      - Index fingertip velocity magnitude (px/frame, normalized)
      - Index fingertip x, y, z (normalized camera coords)
      - Thumb tip x, y, z
      - Mean velocity of all 21 fingertips

    The window is maintained as a rolling deque; features are averaged across
    the window to produce a single vector passed to the classifier.

    Args:
        window_frames: Number of frames in the rolling window.
    """

    FEATURE_DIM: int = 7

    def __init__(self, window_frames: int = _WINDOW_FRAMES) -> None:
        self._window: deque[HandLandmarks] = deque(maxlen=window_frames)

    def update(self, hand: Optional[HandLandmarks]) -> None:
        """Push the latest hand observation into the rolling window.

        Args:
            hand: Current frame landmarks, or None if no hand detected.
        """
        if hand is not None:
            self._window.append(hand)

    def extract(self) -> np.ndarray:
        """Compute the feature vector from the current window.

        Returns:
            Float32 array of shape (FEATURE_DIM,). Returns zeros when the
            window is empty.
        """
        if len(self._window) < 2:
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

        frames = list(self._window)
        # Index fingertip positions across window
        index_positions = np.array(
            [f.landmarks[INDEX_FINGERTIP] for f in frames], dtype=np.float32
        )
        # Velocity: frame-to-frame displacement in xy
        deltas = np.diff(index_positions[:, :2], axis=0)
        velocities = np.linalg.norm(deltas, axis=1)
        mean_index_vel = float(velocities.mean())

        # All-fingertip mean velocity
        all_positions = np.array([f.landmarks for f in frames], dtype=np.float32)
        all_deltas = np.diff(all_positions, axis=0)
        all_vel_mags = np.linalg.norm(all_deltas[:, :, :2], axis=2)
        mean_all_vel = float(all_vel_mags.mean())

        latest = frames[-1].landmarks
        index_xyz = latest[INDEX_FINGERTIP]
        thumb_xyz = latest[THUMB_TIP]

        return np.array(
            [
                mean_index_vel,
                index_xyz[0], index_xyz[1], index_xyz[2],
                thumb_xyz[0], thumb_xyz[1],
                mean_all_vel,
            ],
            dtype=np.float32,
        )

    def reset(self) -> None:
        """Clear the rolling window."""
        self._window.clear()
