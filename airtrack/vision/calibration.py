"""4-point homography calibration: maps distorted webcam pixels to keyboard coords."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from airtrack.config import CALIBRATION_POINTS, CALIBRATION_SAVE_PATH

logger = logging.getLogger(__name__)

# Landmark indices for the four fingertips used during calibration
CALIBRATION_LANDMARK_INDICES: tuple[int, ...] = (4, 8, 12, 16)  # thumb, index, middle, ring


class CalibrationError(Exception):
    """Raised when calibration data is invalid or insufficient."""


class HomographyCalibration:
    """Computes and applies a perspective homography from camera to keyboard space.

    The user taps each of the four keyboard corners in order; the resulting
    homography H maps any camera-space point to a normalized [0,1]×[0,1]
    keyboard coordinate.

    Args:
        save_path: Path used to persist/restore calibration data.
    """

    def __init__(self, save_path: str = CALIBRATION_SAVE_PATH) -> None:
        self._save_path = Path(save_path)
        self._H: Optional[np.ndarray] = None
        self._src_points: list[tuple[float, float]] = []

    @property
    def is_calibrated(self) -> bool:
        """True when a valid homography matrix has been computed."""
        return self._H is not None

    def add_point(self, x: float, y: float) -> bool:
        """Record one calibration tap in camera pixel coordinates.

        Args:
            x: Horizontal pixel coordinate.
            y: Vertical pixel coordinate.

        Returns:
            True when all four points have been collected and H is computed.
        """
        self._src_points.append((x, y))
        if len(self._src_points) == CALIBRATION_POINTS:
            self._compute()
            return True
        return False

    def _compute(self) -> None:
        """Fit the homography from the four collected source points."""
        src = np.array(self._src_points, dtype=np.float32)
        # Canonical unit-square destination (top-left, top-right, bottom-right, bottom-left)
        dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        self._H, _ = cv2.findHomography(src, dst)
        logger.info("Homography computed:\n%s", self._H)

    def transform(self, x: float, y: float) -> tuple[float, float]:
        """Map a camera-space pixel to normalized keyboard coordinates.

        Args:
            x: Camera pixel x.
            y: Camera pixel y.

        Returns:
            (kx, ky) in [0, 1] × [0, 1] keyboard space.

        Raises:
            CalibrationError: If calibration has not been performed.
        """
        if self._H is None:
            raise CalibrationError("Calibration not completed.")
        pt = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self._H)
        kx, ky = float(transformed[0, 0, 0]), float(transformed[0, 0, 1])
        return kx, ky

    def save(self) -> None:
        """Persist the homography matrix to disk."""
        if self._H is None:
            raise CalibrationError("Nothing to save — calibration incomplete.")
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self._save_path, H=self._H)
        logger.info("Calibration saved to %s", self._save_path)

    def load(self) -> None:
        """Load a previously saved homography from disk.

        Raises:
            FileNotFoundError: If the calibration file does not exist.
        """
        if not self._save_path.exists():
            raise FileNotFoundError(f"No calibration at {self._save_path}")
        data = np.load(self._save_path)
        self._H = data["H"]
        logger.info("Calibration loaded from %s", self._save_path)

    def reset(self) -> None:
        """Clear all calibration state."""
        self._H = None
        self._src_points = []
