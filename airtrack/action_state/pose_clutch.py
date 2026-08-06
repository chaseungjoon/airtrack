"""Modality B — Pose clutch: arms GESTURE mode on a deliberate hand pose.

Replaces the v1 acoustic profiler. TYPING vs GESTURE is decided from two
noise-immune signals:

  1. Keystroke lockout — any key event within the lockout window forces
     TYPING unconditionally (Modality A supplies ``seconds_since_last_key``).
  2. Deliberate pose hold — GESTURE arms only after a recognised gesture pose
     (a clean 1–4 extended-finger count from the Finger Pose Estimator) is
     held stably, with an unchanged count, for the hold duration.

A typing hand keeps all fingers semi-curled over home row and never sustains
a clean extended pose, so false activations while typing are structurally
rare rather than threshold-tuned.

State machine:  TYPING --stable pose--> ARMING --held hold_ms--> GESTURE
                GESTURE --pose lost release_ms OR keystroke--> TYPING
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Optional

from airtrack.config import (
    CLUTCH_HOLD_MS,
    CLUTCH_KEYSTROKE_LOCKOUT_MS,
    CLUTCH_RELEASE_MS,
)


class ClutchState(Enum):
    TYPING = "TYPING"
    ARMING = "ARMING"
    GESTURE = "GESTURE"


class PoseClutch:
    """Hysteresis state machine gating the gesture pipeline.

    Args:
        hold_ms: How long a stable pose must be held before GESTURE arms.
        release_ms: How long the pose must be lost before GESTURE releases.
        keystroke_lockout_ms: Keystroke recency that forces TYPING.
    """

    def __init__(
        self,
        hold_ms: float = CLUTCH_HOLD_MS,
        release_ms: float = CLUTCH_RELEASE_MS,
        keystroke_lockout_ms: float = CLUTCH_KEYSTROKE_LOCKOUT_MS,
    ) -> None:
        self._hold_sec = hold_ms / 1000.0
        self._release_sec = release_ms / 1000.0
        self._lockout_sec = keystroke_lockout_ms / 1000.0
        self._state = ClutchState.TYPING
        self._armed_count: Optional[int] = None
        self._arming_since: float = 0.0
        self._pose_lost_since: Optional[float] = None

    @property
    def state(self) -> ClutchState:
        return self._state

    @property
    def is_gesture_active(self) -> bool:
        return self._state is ClutchState.GESTURE

    @property
    def armed_count(self) -> Optional[int]:
        """Finger count the clutch armed with (None outside ARMING/GESTURE)."""
        return self._armed_count

    def update(
        self,
        pose_count: Optional[int],
        seconds_since_last_key: float,
        now: Optional[float] = None,
    ) -> ClutchState:
        """Advance the state machine by one frame.

        Args:
            pose_count: Smoothed extended-finger count (1–4 arms; 0/None does not).
            seconds_since_last_key: Time since the most recent keystroke.
            now: Monotonic timestamp override (for tests).

        Returns:
            The clutch state after this frame.
        """
        t = time.monotonic() if now is None else now
        valid_pose = pose_count is not None and 1 <= pose_count <= 4

        if seconds_since_last_key < self._lockout_sec:
            self._to_typing()
            return self._state

        if self._state is ClutchState.TYPING:
            if valid_pose:
                self._state = ClutchState.ARMING
                self._armed_count = pose_count
                self._arming_since = t

        elif self._state is ClutchState.ARMING:
            if not valid_pose or pose_count != self._armed_count:
                self._to_typing()
                if valid_pose:  # count changed mid-arm: restart with new count
                    self._state = ClutchState.ARMING
                    self._armed_count = pose_count
                    self._arming_since = t
            elif t - self._arming_since >= self._hold_sec:
                self._state = ClutchState.GESTURE
                self._pose_lost_since = None

        elif self._state is ClutchState.GESTURE:
            if valid_pose:
                self._pose_lost_since = None
                self._armed_count = pose_count  # track count changes mid-gesture
            else:
                if self._pose_lost_since is None:
                    self._pose_lost_since = t
                elif t - self._pose_lost_since >= self._release_sec:
                    self._to_typing()

        return self._state

    def reset(self) -> None:
        self._to_typing()

    def _to_typing(self) -> None:
        self._state = ClutchState.TYPING
        self._armed_count = None
        self._pose_lost_since = None
