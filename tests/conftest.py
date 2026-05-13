"""Shared pytest fixtures for AirTrack tests."""

from __future__ import annotations

import numpy as np
import pytest

from airtrack.ml.feature_extractor import FeatureExtractor
from airtrack.ml.state_classifier import HybridStateClassifier
from airtrack.state.mode_manager import ModeManager
from airtrack.vision.calibration import HomographyCalibration
from tests.fakes import (
    FakeCursorController,
    FakeHapticFeedback,
    FakeHandTracker,
    make_hand,
)


@pytest.fixture
def feature_extractor() -> FeatureExtractor:
    return FeatureExtractor()


@pytest.fixture
def classifier() -> HybridStateClassifier:
    return HybridStateClassifier(model_path=None)


@pytest.fixture
def fake_haptic() -> FakeHapticFeedback:
    return FakeHapticFeedback()


@pytest.fixture
def fake_cursor() -> FakeCursorController:
    return FakeCursorController()


@pytest.fixture
def mode_manager(fake_haptic: FakeHapticFeedback) -> ModeManager:
    """ModeManager with real classifier and fake haptic output."""
    return ModeManager(haptic=fake_haptic)


@pytest.fixture
def calibration(tmp_path: object) -> HomographyCalibration:
    """A freshly calibrated homography using the four unit-square corners."""
    cal = HomographyCalibration(save_path=str(tmp_path) + "/cal.npz")  # type: ignore[operator]
    for x, y in [(0, 0), (640, 0), (640, 480), (0, 480)]:
        cal.add_point(float(x), float(y))
    return cal


@pytest.fixture
def still_hand():
    """Returns a callable that produces a stationary hand at (0.5, 0.5)."""
    return lambda: make_hand(0.5, 0.5)


@pytest.fixture
def moving_hand_sequence() -> list:
    """A sequence of hand frames where the index finger sweeps left to right."""
    return [make_hand(i * 0.05, 0.5) for i in range(20)]
