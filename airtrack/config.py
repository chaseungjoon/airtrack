"""Central configuration for AirTrack. All tuneable parameters live here."""

from typing import Final

# --- Camera ---
CAMERA_INDEX: Final[int] = 0
FRAME_WIDTH: Final[int] = 1280
FRAME_HEIGHT: Final[int] = 720
TARGET_FPS: Final[int] = 60

# --- MediaPipe ---
MAX_NUM_HANDS: Final[int] = 2
MIN_DETECTION_CONFIDENCE: Final[float] = 0.7
MIN_TRACKING_CONFIDENCE: Final[float] = 0.5

# --- State Machine ---
# Keystroke rate (keys/sec) above which typing mode is assumed
TYPING_KEYSTROKE_THRESHOLD: Final[float] = 2.0
# Fingertip velocity in normalised [0,1] coords per frame above which gesture mode is triggered
# 0.015 ≈ 19 px/frame at 1280 wide — a brisk lateral sweep
GESTURE_VELOCITY_THRESHOLD: Final[float] = 0.015
# Rolling window duration in seconds used for velocity / keystroke rate
WINDOW_DURATION_SEC: Final[float] = 0.2

# --- Calibration ---
# Number of homography corner points
CALIBRATION_POINTS: Final[int] = 4
CALIBRATION_SAVE_PATH: Final[str] = "models/calibration.npz"

# --- Haptic ---
HAPTIC_PATTERN_GENERIC: Final[str] = "NSHapticFeedbackPatternGeneric"
HAPTIC_PATTERN_ALIGNMENT: Final[str] = "NSHapticFeedbackPatternAlignment"
HAPTIC_PATTERN_LEVEL_CHANGE: Final[str] = "NSHapticFeedbackPatternLevelChange"

# --- Online Learning ---
ONLINE_LEARNING_RATE: Final[float] = 0.01
MIN_CORRECTION_SAMPLES: Final[int] = 5
