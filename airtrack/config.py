"""Central configuration for AirTrack. All tuneable parameters live here."""

from pathlib import Path
from typing import Final

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# --- Camera ---
CAMERA_INDEX: Final[int] = 0
FRAME_WIDTH: Final[int] = 1280
FRAME_HEIGHT: Final[int] = 720
TARGET_FPS: Final[int] = 60

# --- MediaPipe ---
MAX_NUM_HANDS: Final[int] = 2
MIN_DETECTION_CONFIDENCE: Final[float] = 0.7
MIN_TRACKING_CONFIDENCE: Final[float] = 0.5
# HandLandmarker model bundle (Tasks API); auto-downloaded on first use
HAND_LANDMARKER_MODEL_PATH: Final[str] = str(_PROJECT_ROOT / "models" / "hand_landmarker.task")
HAND_LANDMARKER_MODEL_URL: Final[str] = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# --- State Machine ---
# Keystroke rate (keys/sec) above which typing mode is assumed
TYPING_KEYSTROKE_THRESHOLD: Final[float] = 2.0
# Fingertip velocity in normalised [0,1] coords per frame above which gesture mode is triggered
# 0.015 ≈ 19 px/frame at 1280 wide — a brisk lateral sweep
GESTURE_VELOCITY_THRESHOLD: Final[float] = 0.015
# Rolling window duration in seconds used for velocity / keystroke rate
WINDOW_DURATION_SEC: Final[float] = 0.2

# --- Finger Pose (Module 3, Modality A) ---
# Interior PIP-joint angle (degrees) above which a finger counts as extended
POSE_EXTENSION_ANGLE_DEG: Final[float] = 140.0
# Interior IP-joint angle for the thumb (thumb rarely fully straightens)
POSE_THUMB_EXTENSION_ANGLE_DEG: Final[float] = 150.0
# Frames of majority-vote smoothing applied to the finger count
POSE_SMOOTHING_FRAMES: Final[int] = 5

# --- Pose Clutch (Module 1, Modality B) ---
# A stable gesture pose must be held this long before GESTURE mode arms
CLUTCH_HOLD_MS: Final[float] = 120.0
# Pose must be lost this long before GESTURE mode releases back to TYPING
CLUTCH_RELEASE_MS: Final[float] = 200.0
# Any keystroke within this window forces TYPING unconditionally
CLUTCH_KEYSTROKE_LOCKOUT_MS: Final[float] = 150.0

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
