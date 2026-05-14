"""Module 3 fusion: combines bbox morphology, velocity grouping, and sonar
to produce active_fingers (1–4).

Late fusion: each modality casts a weighted vote; majority wins.
Weights reflect empirical reliability (sonar > velocity_grouping > bbox).
When a modality is unavailable (no vision, no audio), its weight is dropped.
"""

from __future__ import annotations

import logging
from typing import Optional

from airtrack.shared.hand_tracker import HandLandmarks
from airtrack.contact.bbox_morphology import BboxMorphology
from airtrack.contact.velocity_grouping import VelocityGrouping
from airtrack.contact.sonar import SonarTransceiver

logger = logging.getLogger(__name__)

# Modality weights: sonar is most reliable when SNR is good
_WEIGHT_BBOX = 1.0
_WEIGHT_VELOCITY = 1.5
_WEIGHT_SONAR = 2.0

# Sonar RMS baseline — frames below this are treated as "no contact"
_SONAR_MIN_RMS = 1e-5


class ContactEstimator:
    """Fuses three modalities to estimate active_fingers (1–4).

    Args:
        sonar: Running SonarTransceiver instance (optional; disables Modality C if None).
        sonar_thresholds: Per-class RMS thresholds [thr_1f, thr_2f, thr_3f] where
            thr_Nf is the upper RMS boundary for N fingers (monotonically increasing).
            Defaults to None (sonar used only for relative comparison).
    """

    def __init__(
        self,
        sonar: Optional[SonarTransceiver] = None,
        sonar_thresholds: Optional[list[float]] = None,
    ) -> None:
        self._bbox = BboxMorphology()
        self._vel_group = VelocityGrouping()
        self._sonar = sonar
        self._sonar_thr = sonar_thresholds  # [thr_1f, thr_2f, thr_3f]

    def estimate(self, hand: Optional[HandLandmarks]) -> Optional[int]:
        """Return active finger count (1–4) or None if confidence is too low.

        Args:
            hand: Current frame landmarks from the primary hand.
        """
        votes: dict[int, float] = {}

        def _cast(n: Optional[int], weight: float) -> None:
            if n is not None:
                votes[n] = votes.get(n, 0.0) + weight

        _cast(self._bbox.estimate(hand), _WEIGHT_BBOX)
        _cast(self._vel_group.estimate(hand), _WEIGHT_VELOCITY)
        _cast(self._sonar_estimate(), _WEIGHT_SONAR)

        if not votes:
            return None

        winner = max(votes, key=lambda k: votes[k])
        total_weight = sum(votes.values())
        confidence = votes[winner] / total_weight

        # Require majority confidence
        if confidence < 0.4:
            return None

        return winner

    def _sonar_estimate(self) -> Optional[int]:
        """Estimate finger count from sonar cross-section energy."""
        if self._sonar is None:
            return None

        frames = self._sonar.get_frames(n=10)
        if not frames:
            return None

        import numpy as np
        rms = float(np.mean([f.doppler_rms for f in frames]))

        if rms < _SONAR_MIN_RMS:
            return None

        if self._sonar_thr and len(self._sonar_thr) >= 3:
            if rms < self._sonar_thr[0]:
                return 1
            if rms < self._sonar_thr[1]:
                return 2
            if rms < self._sonar_thr[2]:
                return 3
            return 4

        return None  # thresholds not calibrated yet

    def reset(self) -> None:
        self._bbox.reset()
        self._vel_group.reset()
