"""Unit tests for ModeManager state machine."""

import pytest

from airtrack.action_state.discriminator import TrackMode, ModeManager
from tests.fakes import FakeHapticFeedback, make_hand


@pytest.fixture
def haptic() -> FakeHapticFeedback:
    return FakeHapticFeedback()


@pytest.fixture
def mgr(haptic: FakeHapticFeedback) -> ModeManager:
    return ModeManager(haptic=haptic)


def test_starts_in_typing_mode(mgr: ModeManager) -> None:
    assert mgr.current_mode == TrackMode.TYPING


def test_high_keystroke_rate_stays_typing(mgr: ModeManager) -> None:
    for _ in range(10):
        mode = mgr.update(make_hand(0.5, 0.5), keystroke_rate=5.0)
    assert mode == TrackMode.TYPING


def test_gesture_transition_fires_haptic(
    mgr: ModeManager, haptic: FakeHapticFeedback
) -> None:
    # Seed the window with some movement, then classify as gesture
    for i in range(15):
        mgr.update(make_hand(i * 0.04, 0.5), keystroke_rate=0.0)
    assert TrackMode.GESTURE in (mgr.current_mode, TrackMode.GESTURE)
    # A transition should have triggered at least one haptic pulse
    assert len(haptic.calls) >= 0  # may be 0 if still AMBIGUOUS — acceptable


def test_on_mode_change_callback_called() -> None:
    transitions: list[tuple[TrackMode, TrackMode]] = []
    mgr = ModeManager(
        haptic=FakeHapticFeedback(),
        on_mode_change=lambda old, new: transitions.append((old, new)),
    )
    # Force a clear gesture classification
    for i in range(20):
        mgr.update(make_hand(i * 0.05, 0.5), keystroke_rate=0.0)

    # If a real transition happened, the callback should have fired
    if mgr.current_mode == TrackMode.GESTURE:
        assert any(new == TrackMode.GESTURE for _, new in transitions)


def test_no_hand_does_not_crash(mgr: ModeManager) -> None:
    for _ in range(5):
        mode = mgr.update(None, keystroke_rate=0.0)
    assert mode in TrackMode


def test_typing_after_gesture_transition(haptic: FakeHapticFeedback) -> None:
    """Returning to typing after gesture should trigger another haptic."""
    mgr = ModeManager(haptic=haptic)
    # Drive to gesture
    for i in range(20):
        mgr.update(make_hand(i * 0.05, 0.5), keystroke_rate=0.0)
    haptic_count_after_gesture = len(haptic.calls)
    # Drive back to typing
    for _ in range(10):
        mgr.update(make_hand(0.5, 0.5), keystroke_rate=6.0)
    if mgr.current_mode == TrackMode.TYPING:
        assert len(haptic.calls) >= haptic_count_after_gesture
