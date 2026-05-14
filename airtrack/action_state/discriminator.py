"""Module 1 fusion: combines keystroke dynamics, acoustic profiling, and bi-manual
kinematics to produce the binary is_gesture_active signal.

Merges HybridStateClassifier + ModeManager from the old ml/state_classifier.py
and state/mode_manager.py into a single module-local unit.
"""

from __future__ import annotations

import enum
import logging
from typing import Callable, Optional

import numpy as np

from airtrack.config import TYPING_KEYSTROKE_THRESHOLD, GESTURE_VELOCITY_THRESHOLD
from airtrack.shared.hand_tracker import HandLandmarks
from airtrack.shared.feedback import HapticFeedback
from airtrack.action_state.bimanual import FeatureExtractor

logger = logging.getLogger(__name__)


class TrackMode(enum.Enum):
    """Operating mode of the AirTrack state machine."""
    TYPING = "typing"
    GESTURE = "gesture"
    AMBIGUOUS = "ambiguous"


class HybridStateClassifier:
    """Two-stage classifier: heuristic gate → neural fallback for ambiguous states.

    Stage 1 — heuristic gate:
      * keystroke_rate > TYPING_KEYSTROKE_THRESHOLD  →  TYPING
      * keystroke_rate ≈ 0 AND velocity > threshold  →  GESTURE
      * otherwise                                     →  AMBIGUOUS

    Stage 2 — neural fallback (optional TCN loaded from disk):
      * GestureClassifierTCN consuming the 7-dim feature vector.

    Args:
        model_path: Path to a TorchScript .pt file. None = heuristics only.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model = self._load_model(model_path)

    def _load_model(self, path: Optional[str]) -> Optional[object]:
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

    def classify(self, feature_vector: np.ndarray, keystroke_rate: float) -> TrackMode:
        """Classify the current state.

        Args:
            feature_vector: Output of FeatureExtractor.extract() — 7-dim float32.
            keystroke_rate: Keystrokes per second over the recent window.
        """
        velocity = float(feature_vector[0])

        if keystroke_rate > TYPING_KEYSTROKE_THRESHOLD:
            return TrackMode.TYPING
        if keystroke_rate < 0.5 and velocity > GESTURE_VELOCITY_THRESHOLD:
            return TrackMode.GESTURE

        if self._model is not None:
            return self._neural_classify(feature_vector)

        return TrackMode.AMBIGUOUS

    def _neural_classify(self, features: np.ndarray) -> TrackMode:
        import torch
        with torch.no_grad():
            tensor = torch.from_numpy(features).unsqueeze(0)
            logits = self._model(tensor)  # type: ignore[operator]
            pred = int(torch.argmax(logits, dim=1).item())
        return TrackMode.TYPING if pred == 0 else TrackMode.GESTURE


class ModeManager:
    """Orchestrates the TYPING ↔ GESTURE state machine for one processing cycle.

    Args:
        classifier: HybridStateClassifier instance.
        feature_extractor: Rolling-window FeatureExtractor.
        haptic: HapticFeedback instance (no-ops if None).
        on_mode_change: Optional callback fired with (old_mode, new_mode).
    """

    def __init__(
        self,
        classifier: Optional[HybridStateClassifier] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        haptic: Optional[HapticFeedback] = None,
        on_mode_change: Optional[Callable[[TrackMode, TrackMode], None]] = None,
    ) -> None:
        self._classifier = classifier or HybridStateClassifier()
        self._extractor = feature_extractor or FeatureExtractor()
        self._haptic = haptic or HapticFeedback()
        self._on_mode_change = on_mode_change
        self._current_mode: TrackMode = TrackMode.TYPING

    @property
    def current_mode(self) -> TrackMode:
        return self._current_mode

    def update(
        self,
        hand: Optional[HandLandmarks],
        keystroke_rate: float,
    ) -> TrackMode:
        """Process one frame and return the active mode."""
        self._extractor.update(hand)
        features = self._extractor.extract()
        new_mode = self._classifier.classify(features, keystroke_rate)

        if new_mode != TrackMode.AMBIGUOUS and new_mode != self._current_mode:
            old_mode = self._current_mode
            self._current_mode = new_mode
            logger.info("Mode: %s → %s", old_mode.value, new_mode.value)
            self._haptic.trigger("level_change")
            if self._on_mode_change is not None:
                self._on_mode_change(old_mode, new_mode)

        return self._current_mode
