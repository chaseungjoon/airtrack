"""Webcam capture with configurable resolution and frame rate."""

import cv2
import numpy as np
from typing import Optional, Generator

from airtrack.config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, TARGET_FPS


class CameraCapture:
    """Manages webcam lifecycle and yields BGR frames.

    Args:
        index: OS camera device index.
        width: Desired capture width in pixels.
        height: Desired capture height in pixels.
        fps: Desired capture frame rate.
    """

    def __init__(
        self,
        index: int = CAMERA_INDEX,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        fps: int = TARGET_FPS,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open the camera device and apply settings."""
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {self._index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

    def read(self) -> Optional[np.ndarray]:
        """Read a single BGR frame.

        Returns:
            BGR frame array or None if capture failed.
        """
        if self._cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")
        ret, frame = self._cap.read()
        return frame if ret else None

    def stream(self) -> Generator[np.ndarray, None, None]:
        """Yield frames continuously until the camera is released.

        Yields:
            BGR frame as a numpy array.
        """
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

    def release(self) -> None:
        """Release the camera device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraCapture":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
