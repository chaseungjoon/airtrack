"""Modality A — Extended-finger pose counting via PIP joint angles.

Finger count is read from hand *pose* (which fingers are extended), not from
physical surface contact. For each non-thumb finger the interior angle at the
PIP joint — between vectors PIP→MCP and PIP→TIP in 3D landmark space — is
near 180° when the finger is extended and collapses below ~90° when curled.
The classification is rotation- and view-invariant, so it works with both a
front-facing webcam and a top-down Desk View / mirror rig.

A temporal majority vote over the last N frames stabilises the count against
single-frame landmark jitter.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from airtrack.config import (
    POSE_EXTENSION_ANGLE_DEG,
    POSE_SMOOTHING_FRAMES,
    POSE_THUMB_EXTENSION_ANGLE_DEG,
)
from airtrack.shared.hand_tracker import HandLandmarks

# MediaPipe landmark indices (proximal joint, hinge joint, tip) per finger.
# Non-thumb fingers hinge at the PIP; the thumb hinges at the IP joint.
FINGER_JOINT_INDICES: dict[str, tuple[int, int, int]] = {
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}
THUMB_JOINT_INDICES: tuple[int, int, int] = (2, 3, 4)


@dataclass(frozen=True)
class FingerPose:
    """Per-finger extension state for a single frame."""

    index: bool
    middle: bool
    ring: bool
    pinky: bool
    thumb: bool

    @property
    def count(self) -> int:
        """Number of extended non-thumb fingers (the gesture-grammar count)."""
        return int(self.index) + int(self.middle) + int(self.ring) + int(self.pinky)


def _interior_angle_deg(landmarks: np.ndarray, joints: tuple[int, int, int]) -> float:
    """Interior angle at the hinge joint, in degrees, using 3D coordinates."""
    proximal, hinge, tip = joints
    u = landmarks[proximal] - landmarks[hinge]
    v = landmarks[tip] - landmarks[hinge]
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom < 1e-9:
        return 0.0
    cos_angle = float(np.dot(u, v)) / denom
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def classify_pose(
    hand: HandLandmarks,
    extension_angle_deg: float = POSE_EXTENSION_ANGLE_DEG,
    thumb_extension_angle_deg: float = POSE_THUMB_EXTENSION_ANGLE_DEG,
) -> FingerPose:
    """Classify per-finger extension for one frame of landmarks.

    Args:
        hand: Detected hand landmarks.
        extension_angle_deg: PIP angle above which a non-thumb finger is extended.
        thumb_extension_angle_deg: IP angle above which the thumb is extended.

    Returns:
        FingerPose with a boolean per finger.
    """
    lm = hand.landmarks
    states = {
        name: _interior_angle_deg(lm, joints) >= extension_angle_deg
        for name, joints in FINGER_JOINT_INDICES.items()
    }
    thumb = _interior_angle_deg(lm, THUMB_JOINT_INDICES) >= thumb_extension_angle_deg
    return FingerPose(thumb=thumb, **states)


class PoseFingerCounter:
    """Smoothed extended-finger counter over a rolling frame window.

    Args:
        extension_angle_deg: PIP angle threshold for non-thumb fingers.
        smoothing_frames: Window length for the majority vote.
    """

    def __init__(
        self,
        extension_angle_deg: float = POSE_EXTENSION_ANGLE_DEG,
        smoothing_frames: int = POSE_SMOOTHING_FRAMES,
    ) -> None:
        self._angle_thr = extension_angle_deg
        self._counts: deque[int] = deque(maxlen=max(1, smoothing_frames))
        self._last_pose: Optional[FingerPose] = None

    @property
    def last_pose(self) -> Optional[FingerPose]:
        """Most recent per-frame pose, or None if no hand was seen."""
        return self._last_pose

    def update(self, hand: Optional[HandLandmarks]) -> Optional[int]:
        """Process one frame and return the smoothed finger count.

        Args:
            hand: Latest hand landmarks, or None if no hand visible.

        Returns:
            Majority-vote count of extended non-thumb fingers (0–4), or None
            while no hand is visible.
        """
        if hand is None:
            self.reset()
            return None

        pose = classify_pose(hand, extension_angle_deg=self._angle_thr)
        self._last_pose = pose
        self._counts.append(pose.count)

        most_common = Counter(self._counts).most_common(1)
        return most_common[0][0]

    def reset(self) -> None:
        self._counts.clear()
        self._last_pose = None
