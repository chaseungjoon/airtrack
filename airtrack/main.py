"""AirTrack entry point — wires together all subsystems and runs the main loop."""

from __future__ import annotations

import logging

from airtrack.camera.capture import CameraCapture
from airtrack.vision.hand_tracker import HandTracker
from airtrack.vision.calibration import HomographyCalibration
from airtrack.input.keyboard_listener import KeystrokeMonitor
from airtrack.input.cursor_controller import CursorController
from airtrack.state.mode_manager import ModeManager
from airtrack.ml.state_classifier import TrackMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Placeholder screen dimensions — replaced by actual display query at startup
_DEFAULT_SCREEN_WIDTH = 2560
_DEFAULT_SCREEN_HEIGHT = 1600


def _get_screen_resolution() -> tuple[int, int]:
    """Query the primary display resolution via Quartz."""
    try:
        import Quartz  # type: ignore[import]
        display = Quartz.CGMainDisplayID()
        return (
            Quartz.CGDisplayPixelsWide(display),
            Quartz.CGDisplayPixelsHigh(display),
        )
    except Exception:
        return _DEFAULT_SCREEN_WIDTH, _DEFAULT_SCREEN_HEIGHT


def main() -> None:
    """Run the AirTrack gesture tracking loop."""
    screen_w, screen_h = _get_screen_resolution()
    logger.info("Screen: %dx%d", screen_w, screen_h)

    calibration = HomographyCalibration()
    try:
        calibration.load()
        logger.info("Loaded existing calibration.")
    except FileNotFoundError:
        logger.warning("No calibration found. Run `airtrack-calibrate` first.")

    cursor = CursorController(screen_w, screen_h)
    mode_mgr = ModeManager()

    with CameraCapture() as cam, HandTracker() as tracker, KeystrokeMonitor() as keys:
        logger.info("AirTrack running. Press Ctrl+C to quit.")
        try:
            for frame in cam.stream():
                hands = tracker.process(frame)
                primary_hand = hands[0] if hands else None

                mode = mode_mgr.update(primary_hand, keys.keystroke_rate)

                if mode == TrackMode.GESTURE and primary_hand is not None and calibration.is_calibrated:
                    # Index fingertip → cursor position
                    lm = primary_hand.landmarks[8]  # INDEX_FINGERTIP
                    kx, ky = calibration.transform(lm[0], lm[1])
                    cursor.move_normalized(kx, ky)
        except KeyboardInterrupt:
            logger.info("Shutting down.")


if __name__ == "__main__":
    main()
