"""MediaPipe HandLandmarker wrapper producing normalized 21-point landmark arrays.

mediapipe ≥ 0.10.3x removed the legacy ``mp.solutions`` API, so this wraps the
Tasks API (``HandLandmarker``) in VIDEO mode behind the same interface the rest
of the codebase already uses. The model bundle is downloaded automatically to
``HAND_LANDMARKER_MODEL_PATH`` on first use (~8 MB, one time).
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from typing import NamedTuple

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python import vision

from airtrack.config import (
    HAND_LANDMARKER_MODEL_PATH,
    HAND_LANDMARKER_MODEL_URL,
    MAX_NUM_HANDS,
    MIN_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
)


class HandLandmarks(NamedTuple):
    """Structured result from a single detected hand.

    Attributes:
        landmarks: (21, 3) float32 array of (x, y, z) in normalized [0,1] coords.
        handedness: "Left" or "Right".
    """
    landmarks: np.ndarray
    handedness: str


def _ensure_model(model_path: str = HAND_LANDMARKER_MODEL_PATH) -> str:
    """Download the HandLandmarker model bundle if it is not present."""
    path = Path(model_path)
    if path.exists():
        return model_path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[hand_tracker] Downloading HandLandmarker model to {path} ...")
    tmp = path.with_suffix(".task.part")
    urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, tmp)
    tmp.rename(path)
    print("[hand_tracker] Model downloaded.")
    return model_path


class HandTracker:
    """Wraps the MediaPipe Tasks HandLandmarker for per-frame landmark detection."""

    def __init__(
        self,
        max_hands: int = MAX_NUM_HANDS,
        detection_confidence: float = MIN_DETECTION_CONFIDENCE,
        tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
        model_path: str = HAND_LANDMARKER_MODEL_PATH,
    ) -> None:
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_ensure_model(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def process(self, bgr_frame: np.ndarray) -> list[HandLandmarks]:
        """Detect hand landmarks in a BGR frame."""
        rgb = np.ascontiguousarray(bgr_frame[:, :, ::-1])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode requires strictly increasing timestamps
        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)

        output: list[HandLandmarks] = []
        for lm_list, handedness in zip(result.hand_landmarks, result.handedness):
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32
            )
            label = handedness[0].category_name
            output.append(HandLandmarks(landmarks=coords, handedness=label))
        return output

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
