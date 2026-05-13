"""State machine that coordinates mode transitions and side effects."""

from __future__ import annotations

import logging
from typing import Callable, Optional

from airtrack.ml.state_classifier import HybridStateClassifier, TrackMode
from airtrack.ml.feature_extractor import FeatureExtractor
from airtrack.vision.hand_tracker import HandLandmarks
from airtrack.feedback.haptic import HapticFeedback

logger = logging.getLogger(__name__)


class ModeManager:
    """Orchestrates the TYPING ↔ GESTURE state machine for one processing cycle.

    Each call to ``update`` takes the latest hand observation and keystroke
    rate, updates internal state, fires haptic feedback on transitions, and
    returns the active TrackMode.

    Args:
        classifier: Hybrid state classifier instance.
        feature_extractor: Rolling window feature extractor.
        haptic: Haptic feedback interface.
        on_mode_change: Optional callback invoked with (old_mode, new_mode)
            on every transition.
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
        """The last resolved operating mode."""
        return self._current_mode

    def update(
        self,
        hand: Optional[HandLandmarks],
        keystroke_rate: float,
    ) -> TrackMode:
        """Process one frame and return the active mode.

        Args:
            hand: Latest hand landmarks (None when no hand detected).
            keystroke_rate: Current keystrokes-per-second rate.

        Returns:
            The resulting TrackMode after classification.
        """
        self._extractor.update(hand)
        features = self._extractor.extract()
        new_mode = self._classifier.classify(features, keystroke_rate)

        if new_mode != TrackMode.AMBIGUOUS and new_mode != self._current_mode:
            old_mode = self._current_mode
            self._current_mode = new_mode
            logger.info("Mode transition: %s → %s", old_mode.value, new_mode.value)
            self._haptic.trigger("level_change")
            if self._on_mode_change is not None:
                self._on_mode_change(old_mode, new_mode)

        return self._current_mode
