"""Unit tests for HybridStateClassifier heuristic gate."""

import numpy as np

from airtrack.ml.state_classifier import HybridStateClassifier, TrackMode
from airtrack.ml.feature_extractor import FeatureExtractor


def _feature(velocity: float = 0.0) -> np.ndarray:
    vec = np.zeros(FeatureExtractor.FEATURE_DIM, dtype=np.float32)
    vec[0] = velocity
    vec[6] = velocity
    return vec


def test_high_keystroke_rate_gives_typing() -> None:
    clf = HybridStateClassifier()
    mode = clf.classify(_feature(velocity=20.0), keystroke_rate=5.0)
    assert mode == TrackMode.TYPING


def test_zero_keystroke_high_velocity_gives_gesture() -> None:
    clf = HybridStateClassifier()
    mode = clf.classify(_feature(velocity=15.0), keystroke_rate=0.0)
    assert mode == TrackMode.GESTURE


def test_ambiguous_without_neural_model() -> None:
    clf = HybridStateClassifier(model_path=None)
    mode = clf.classify(_feature(velocity=5.0), keystroke_rate=1.0)
    assert mode == TrackMode.AMBIGUOUS
