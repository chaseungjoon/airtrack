"""Unit tests for FeatureExtractor."""

import numpy as np
import pytest

from airtrack.ml.feature_extractor import FeatureExtractor, FeatureVector
from airtrack.vision.hand_tracker import HandLandmarks


def _make_hand(x: float = 0.5, y: float = 0.5) -> HandLandmarks:
    lm = np.zeros((21, 3), dtype=np.float32)
    lm[8] = [x, y, 0.0]
    lm[4] = [x - 0.05, y - 0.05, 0.0]
    return HandLandmarks(landmarks=lm, handedness="Right")


def test_returns_zeros_on_empty_window() -> None:
    ext = FeatureExtractor()
    vec = ext.extract()
    assert vec.shape == (FeatureExtractor.FEATURE_DIM,)
    assert np.all(vec == 0.0)


def test_nonzero_after_two_frames() -> None:
    ext = FeatureExtractor()
    ext.update(_make_hand(0.3, 0.3))
    ext.update(_make_hand(0.5, 0.5))
    vec = ext.extract()
    assert vec[0] > 0.0, "velocity should be nonzero after movement"


def test_reset_clears_window() -> None:
    ext = FeatureExtractor()
    ext.update(_make_hand())
    ext.update(_make_hand(0.8, 0.8))
    ext.reset()
    vec = ext.extract()
    assert np.all(vec == 0.0)


def test_feature_dim() -> None:
    ext = FeatureExtractor()
    for i in range(5):
        ext.update(_make_hand(i * 0.1, i * 0.1))
    assert ext.extract().shape == (FeatureExtractor.FEATURE_DIM,)
