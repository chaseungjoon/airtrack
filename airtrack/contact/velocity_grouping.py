"""Modality B — Synchronized velocity grouping via cosine similarity.

Evaluates whether adjacent fingertips share the same velocity vector within
a tolerance margin. Fingers that move together (cosine similarity > threshold)
are counted as a single swipe cluster, regardless of Z-depth hover.

This resolves the core Z-axis ambiguity: a finger 2 mm above the keyboard
may look like it's touching from vision, but if it doesn't move in sync with
the touching fingers it is excluded from the cluster count.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from airtrack.shared.hand_tracker import HandLandmarks

FINGERTIP_INDICES: tuple[int, ...] = (4, 8, 12, 16, 20)

_COSINE_SIMILARITY_THRESHOLD = 0.95  # ~18° tolerance
_MIN_VELOCITY = 0.003                 # filter near-static fingertips
_WINDOW_FRAMES = 3                    # smooth over N frames before comparing


class VelocityGrouping:
    """Counts synchronized fingertips to estimate active finger count.

    Args:
        similarity_threshold: Cosine similarity threshold to group two fingertips.
        window_frames: Frames over which velocity is averaged before comparison.
    """

    def __init__(
        self,
        similarity_threshold: float = _COSINE_SIMILARITY_THRESHOLD,
        window_frames: int = _WINDOW_FRAMES,
    ) -> None:
        self._thr = similarity_threshold
        self._history: deque[np.ndarray] = deque(maxlen=window_frames + 1)

    def estimate(self, hand: Optional[HandLandmarks]) -> Optional[int]:
        """Return synchronized finger count (1–4) or None if undetermined."""
        if hand is None:
            self._history.clear()
            return None

        tips = hand.landmarks[list(FINGERTIP_INDICES), :2]  # (5, 2)
        self._history.append(tips)

        if len(self._history) < 2:
            return None

        # Mean velocity vector per fingertip over the window
        frames = np.stack(list(self._history))          # (T, 5, 2)
        velocities = np.diff(frames, axis=0).mean(0)    # (5, 2) mean delta

        speeds = np.linalg.norm(velocities, axis=1)     # (5,)
        active_mask = speeds > _MIN_VELOCITY

        if active_mask.sum() == 0:
            return None

        active_vels = velocities[active_mask]            # (k, 2)
        norms = np.linalg.norm(active_vels, axis=1, keepdims=True)
        unit_vels = active_vels / np.maximum(norms, 1e-10)

        # Greedy clustering: seed with first active fingertip
        groups: list[list[int]] = [[0]]
        for i in range(1, len(unit_vels)):
            placed = False
            for g in groups:
                ref = unit_vels[g[0]]
                sim = float(np.dot(unit_vels[i], ref))
                if sim >= self._thr:
                    g.append(i)
                    placed = True
                    break
            if not placed:
                groups.append([i])

        # Largest synchronized group = active finger count
        count = max(len(g) for g in groups)
        return min(count, 4)

    def reset(self) -> None:
        self._history.clear()
