"""Hybrid state classifier: heuristic gate + lightweight neural fallback.

The heuristic gate is fast and covers obvious cases (rapid keystrokes →
typing, high velocity + no keys → gesture). The neural model handles
ambiguous in-between states.
"""

from __future__ import annotations

import enum
import logging
from typing import Optional

import numpy as np

from airtrack.config import TYPING_KEYSTROKE_THRESHOLD, GESTURE_VELOCITY_THRESHOLD

logger = logging.getLogger(__name__)


class TrackMode(enum.Enum):
    """Operating mode of the AirTrack state machine."""
    TYPING = "typing"
    GESTURE = "gesture"
    AMBIGUOUS = "ambiguous"


class HybridStateClassifier:
    """Two-stage classifier: heuristic → neural for ambiguous states.

    Stage 1 — heuristic gate:
      * keystroke_rate > TYPING_KEYSTROKE_THRESHOLD  →  TYPING
      * keystroke_rate ≈ 0 AND velocity > GESTURE_VELOCITY_THRESHOLD  →  GESTURE
      * otherwise  →  ambiguous

    Stage 2 — neural model (placeholder, wired in Phase 2):
      * A TCN/LSTM loaded from CoreML or PyTorch that consumes the feature
        vector and outputs a probability over {TYPING, GESTURE}.

    Args:
        model_path: Optional path to a serialized neural model. When None,
            only the heuristic gate is used.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model = self._load_model(model_path)

    def _load_model(self, path: Optional[str]) -> Optional[object]:
        """Load the neural model if a path is provided."""
        if path is None:
            return None
        try:
            import torch
            model = torch.jit.load(path)
            model.eval()
            logger.info("Neural state model loaded from %s", path)
            return model
        except Exception as exc:
            logger.warning("Failed to load neural model (%s); heuristics only.", exc)
            return None

    def classify(
        self,
        feature_vector: np.ndarray,
        keystroke_rate: float,
    ) -> TrackMode:
        """Classify the current state.

        Args:
            feature_vector: Output of FeatureExtractor.extract().
            keystroke_rate: Keystrokes per second over the recent window.

        Returns:
            The inferred TrackMode.
        """
        # Stage 1: heuristic gate
        velocity = float(feature_vector[0])

        if keystroke_rate > TYPING_KEYSTROKE_THRESHOLD:
            return TrackMode.TYPING
        if keystroke_rate < 0.5 and velocity > GESTURE_VELOCITY_THRESHOLD:
            return TrackMode.GESTURE

        # Stage 2: neural fallback
        if self._model is not None:
            return self._neural_classify(feature_vector)

        return TrackMode.AMBIGUOUS

    def _neural_classify(self, features: np.ndarray) -> TrackMode:
        """Run the neural model on a feature vector.

        Args:
            features: Float32 array of shape (FEATURE_DIM,).

        Returns:
            TrackMode based on the model's argmax prediction.
        """
        import torch
        with torch.no_grad():
            tensor = torch.from_numpy(features).unsqueeze(0)
            logits = self._model(tensor)  # type: ignore[operator]
            pred = int(torch.argmax(logits, dim=1).item())
        return TrackMode.TYPING if pred == 0 else TrackMode.GESTURE
