"""Interactive 4-point calibration utility.

Usage:
    python scripts/calibrate.py

The user taps the index finger to each of the four keyboard corners when
prompted. A homography matrix is computed and saved to models/calibration.npz.
"""

from __future__ import annotations

import cv2
import logging

from airtrack.shared.camera_capture import CameraCapture
from airtrack.shared.hand_tracker import HandTracker
from airtrack.shared.calibration import HomographyCalibration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CORNER_LABELS = [
    "TOP-LEFT keyboard corner",
    "TOP-RIGHT keyboard corner",
    "BOTTOM-RIGHT keyboard corner",
    "BOTTOM-LEFT keyboard corner",
]


def run_calibration() -> None:
    """Launch the interactive calibration loop."""
    calibration = HomographyCalibration()
    point_idx = 0

    print("=== AirTrack Calibration ===")
    print(f"Point your index finger at the {CORNER_LABELS[point_idx]}")
    print("Press SPACE to record each point, Q to quit.")

    with CameraCapture() as cam, HandTracker() as tracker:
        for frame in cam.stream():
            hands = tracker.process(frame)
            display = frame.copy()

            if hands:
                lm = hands[0].landmarks[8]  # index fingertip
                h, w = frame.shape[:2]
                cx, cy = int(lm[0] * w), int(lm[1] * h)
                cv2.circle(display, (cx, cy), 12, (0, 255, 0), -1)

            label = CORNER_LABELS[point_idx] if point_idx < 4 else "Done!"
            cv2.putText(display, f"Point {point_idx + 1}/4: {label}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("AirTrack Calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" ") and hands:
                lm = hands[0].landmarks[8]
                h, w = frame.shape[:2]
                px, py = lm[0] * w, lm[1] * h
                done = calibration.add_point(px, py)
                logger.info("Recorded point %d: (%.1f, %.1f)", point_idx + 1, px, py)
                point_idx += 1
                if done:
                    calibration.save()
                    print("Calibration complete and saved.")
                    break
                print(f"Point your index finger at the {CORNER_LABELS[point_idx]}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_calibration()
