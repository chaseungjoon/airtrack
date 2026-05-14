"""MediaPipe Hands wrapper producing normalized 21-point landmark arrays."""

from __future__ import annotations

import numpy as np
import mediapipe as mp
from typing import NamedTuple

from airtrack.config import MAX_NUM_HANDS, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE


class HandLandmarks(NamedTuple):
    """Structured result from a single detected hand.

    Attributes:
        landmarks: (21, 3) float32 array of (x, y, z) in normalized [0,1] coords.
        handedness: "Left" or "Right".
    """
    landmarks: np.ndarray
    handedness: str


class HandTracker:
    """Wraps MediaPipe Hands for per-frame landmark detection."""

    def __init__(
        self,
        max_hands: int = MAX_NUM_HANDS,
        detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
    ) -> None:
        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

    def process(self, bgr_frame: np.ndarray) -> list[HandLandmarks]:
        """Detect hand landmarks in a BGR frame."""
        rgb = bgr_frame[:, :, ::-1]
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return []
        output: list[HandLandmarks] = []
        for lm_list, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in lm_list.landmark], dtype=np.float32
            )
            label = handedness.classification[0].label
            output.append(HandLandmarks(landmarks=coords, handedness=label))
        return output

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
