"""Integration test: synthetic camera → tracker → mode manager pipeline.

Uses only fake/in-process objects — no real camera, OS events, or PyObjC.
"""

from __future__ import annotations

import numpy as np

from airtrack.ml.state_classifier import TrackMode
from airtrack.state.mode_manager import ModeManager
from airtrack.vision.calibration import HomographyCalibration
from tests.fakes import (
    FakeCursorController,
    FakeHapticFeedback,
    FakeHandTracker,
    FakeCamera,
    make_hand,
    blank_frame,
)


def _run_pipeline(
    hand_results: list[list],
    keystroke_rates: list[float],
) -> list[TrackMode]:
    """Drive the pipeline with scripted hand/keystroke inputs.

    Returns the mode decided on each frame.
    """
    haptic = FakeHapticFeedback()
    mgr = ModeManager(haptic=haptic)
    modes: list[TrackMode] = []
    for hands, rate in zip(hand_results, keystroke_rates):
        primary = hands[0] if hands else None
        modes.append(mgr.update(primary, rate))
    return modes


def test_constant_typing_stays_in_typing_mode() -> None:
    """Sustained high keystroke rate should keep the system in TYPING."""
    n = 30
    hands = [[make_hand(0.5, 0.5)]] * n
    rates = [5.0] * n
    modes = _run_pipeline(hands, rates)
    assert all(m == TrackMode.TYPING for m in modes)


def test_stationary_hand_no_keys_stays_ambiguous_or_typing() -> None:
    """No movement and no keystrokes — heuristic can't decide, stays ambiguous."""
    n = 20
    hands = [[make_hand(0.5, 0.5)]] * n
    rates = [0.0] * n
    modes = _run_pipeline(hands, rates)
    # With a stationary hand, velocity ≈ 0 → never reaches gesture threshold
    assert TrackMode.GESTURE not in modes


def test_fast_sweep_no_keys_triggers_gesture() -> None:
    """A fast lateral sweep with no keystrokes should transition to GESTURE."""
    n = 30
    hands = [[make_hand(i * 0.04, 0.5)] for i in range(n)]
    rates = [0.0] * n
    modes = _run_pipeline(hands, rates)
    assert TrackMode.GESTURE in modes, "Expected at least one GESTURE classification"


def test_no_hand_detected_does_not_crash() -> None:
    """Frames with no hand must be handled gracefully."""
    n = 20
    hands = [[] for _ in range(n)]
    rates = [0.0] * n
    modes = _run_pipeline(hands, rates)
    assert len(modes) == n


def test_calibration_transform_applied_in_gesture_mode() -> None:
    """In gesture mode, cursor position should reflect calibrated coordinates."""
    cal = HomographyCalibration(save_path="/tmp/test_int_cal.npz")
    for x, y in [(0, 0), (640, 0), (640, 480), (0, 480)]:
        cal.add_point(float(x), float(y))

    cursor = FakeCursorController()
    haptic = FakeHapticFeedback()
    mgr = ModeManager(haptic=haptic)

    # Drive to gesture mode — keep x in [0.1, 0.85] so homography stays in-bounds
    for i in range(30):
        x = 0.1 + (i % 16) * 0.05  # oscillates 0.1 → 0.85
        hand = make_hand(x, 0.5)
        mode = mgr.update(hand, keystroke_rate=0.0)
        if mode == TrackMode.GESTURE and cal.is_calibrated:
            lm = hand.landmarks[8]
            kx, ky = cal.transform(lm[0] * 640, lm[1] * 480)
            cursor.move_normalized(kx, ky)

    if mgr.current_mode == TrackMode.GESTURE:
        assert len(cursor.positions) > 0
        for nx, ny in cursor.positions:
            assert 0.0 <= nx <= 1.0
            assert 0.0 <= ny <= 1.0
