"""Modality B — Heuristic motion parsing with Kalman smoothing.

Converts raw MediaPipe landmark velocities into smooth (dx, dy) cursor deltas,
filtering out micro-jitters from the tracking noise.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from airtrack.shared.hand_tracker import HandLandmarks
from airtrack.config import WINDOW_DURATION_SEC, TARGET_FPS

INDEX_FINGERTIP: int = 8

_WINDOW_FRAMES: int = max(1, int(WINDOW_DURATION_SEC * TARGET_FPS))

# Simple scalar Kalman filter parameters for 1-D position smoothing
_KF_PROCESS_NOISE = 1e-3
_KF_MEASURE_NOISE = 1e-2


class KalmanSmoother1D:
    """Scalar Kalman filter for smoothing a single noisy measurement stream."""

    def __init__(self, q: float = _KF_PROCESS_NOISE, r: float = _KF_MEASURE_NOISE) -> None:
        self._q = q   # process noise covariance
        self._r = r   # measurement noise covariance
        self._x = 0.0  # state estimate
        self._p = 1.0  # error covariance

    def update(self, z: float) -> float:
        # Predict
        self._p += self._q
        # Update
        k = self._p / (self._p + self._r)
        self._x += k * (z - self._x)
        self._p *= (1.0 - k)
        return self._x

    def reset(self, value: float = 0.0) -> None:
        self._x = value
        self._p = 1.0


class MotionParser:
    """Extracts smooth cursor (dx, dy) from a rolling landmark window.

    Args:
        window_frames: History depth for velocity estimation.
        speed_scale: Multiplier applied to the normalized velocity before output.
    """

    def __init__(
        self,
        window_frames: int = _WINDOW_FRAMES,
        speed_scale: float = 3.0,
    ) -> None:
        self._window: deque[np.ndarray] = deque(maxlen=window_frames)
        self._kf_x = KalmanSmoother1D()
        self._kf_y = KalmanSmoother1D()
        self._speed_scale = speed_scale
        self._prev_pos: Optional[np.ndarray] = None

    def update(self, hand: Optional[HandLandmarks]) -> tuple[float, float]:
        """Push a frame and return smoothed (dx, dy) cursor delta.

        Args:
            hand: Latest hand landmarks, or None if no hand detected.

        Returns:
            (dx, dy) in normalized [0,1] units. (0, 0) if no motion.
        """
        if hand is None:
            self._prev_pos = None
            return 0.0, 0.0

        pos = hand.landmarks[INDEX_FINGERTIP, :2].copy()
        self._window.append(pos)

        if self._prev_pos is None:
            self._prev_pos = pos
            return 0.0, 0.0

        raw_dx = float(pos[0] - self._prev_pos[0]) * self._speed_scale
        raw_dy = float(pos[1] - self._prev_pos[1]) * self._speed_scale
        self._prev_pos = pos

        dx = self._kf_x.update(raw_dx)
        dy = self._kf_y.update(raw_dy)
        return dx, dy

    def reset(self) -> None:
        self._window.clear()
        self._prev_pos = None
        self._kf_x.reset()
        self._kf_y.reset()
