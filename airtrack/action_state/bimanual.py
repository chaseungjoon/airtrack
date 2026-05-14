"""Modality C — Asymmetric bi-manual kinematics for TYPING vs GESTURE discrimination.

During typing both hands exhibit high-frequency, low-amplitude symmetrical jitter.
During a gesture, one hand sweeps (high-amplitude, low-frequency) while the other
stays idle — producing a strong asymmetry in velocity variance.

Also exposes FeatureExtractor (the per-hand rolling velocity/position vector)
which is consumed by both the discriminator and the kinematic decoder.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from airtrack.shared.hand_tracker import HandLandmarks
from airtrack.config import WINDOW_DURATION_SEC, TARGET_FPS

INDEX_FINGERTIP: int = 8
THUMB_TIP: int = 4

_WINDOW_FRAMES: int = max(1, int(WINDOW_DURATION_SEC * TARGET_FPS))


class FeatureExtractor:
    """Computes a fixed-length feature vector from a rolling landmark window.

    Features (7 values):
      [0] index fingertip velocity magnitude (normalized)
      [1] index tip x  [2] y  [3] z
      [4] thumb tip x  [5] y
      [6] mean velocity across all 21 landmarks
    """

    FEATURE_DIM: int = 7

    def __init__(self, window_frames: int = _WINDOW_FRAMES) -> None:
        self._window: deque[HandLandmarks] = deque(maxlen=window_frames)

    def update(self, hand: Optional[HandLandmarks]) -> None:
        if hand is not None:
            self._window.append(hand)

    def extract(self) -> np.ndarray:
        if len(self._window) < 2:
            return np.zeros(self.FEATURE_DIM, dtype=np.float32)

        frames = list(self._window)
        index_pos = np.array([f.landmarks[INDEX_FINGERTIP] for f in frames], dtype=np.float32)
        deltas = np.diff(index_pos[:, :2], axis=0)
        mean_index_vel = float(np.linalg.norm(deltas, axis=1).mean())

        all_pos = np.array([f.landmarks for f in frames], dtype=np.float32)
        all_deltas = np.diff(all_pos, axis=0)
        mean_all_vel = float(np.linalg.norm(all_deltas[:, :, :2], axis=2).mean())

        latest = frames[-1].landmarks
        index_xyz = latest[INDEX_FINGERTIP]
        thumb_xyz = latest[THUMB_TIP]

        return np.array(
            [mean_index_vel, index_xyz[0], index_xyz[1], index_xyz[2],
             thumb_xyz[0], thumb_xyz[1], mean_all_vel],
            dtype=np.float32,
        )

    def reset(self) -> None:
        self._window.clear()


class BimanualAnalyzer:
    """Measures velocity asymmetry between left and right hands.

    High asymmetry (one hand sweeping, one idle) → likely GESTURE.
    Low asymmetry (both hands showing typing jitter) → likely TYPING.
    """

    def __init__(self, window_frames: int = _WINDOW_FRAMES) -> None:
        self._left: deque[np.ndarray] = deque(maxlen=window_frames)
        self._right: deque[np.ndarray] = deque(maxlen=window_frames)

    def update(self, hands: list[HandLandmarks]) -> None:
        for h in hands:
            pos = h.landmarks[INDEX_FINGERTIP, :2]
            if h.handedness == "Left":
                self._left.append(pos)
            else:
                self._right.append(pos)

    def asymmetry_score(self) -> float:
        """Returns 0 (symmetric) → 1 (fully asymmetric).

        High score suggests one hand is gesturing while the other is idle.
        """
        def _var(buf: deque[np.ndarray]) -> float:
            if len(buf) < 2:
                return 0.0
            arr = np.stack(list(buf))
            return float(np.var(np.diff(arr, axis=0)))

        left_v = _var(self._left)
        right_v = _var(self._right)
        total = left_v + right_v
        if total < 1e-10:
            return 0.0
        return float(abs(left_v - right_v) / total)
