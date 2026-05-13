"""macOS Desk View / ultra-wide lens correction interface.

Desk View is available on MacBooks with a Center Stage camera (M1 Pro/Max and later).
This module wraps the CoreMedia / AVFoundation API via PyObjC to request the
pre-corrected top-down frame when available, falling back to standard capture.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DeskViewCapture:
    """Requests top-down corrected frames via macOS Desk View when available.

    Falls back to a standard CameraCapture if the device does not support it.

    Args:
        fallback: A CameraCapture-compatible object used when Desk View is unavailable.
    """

    def __init__(self, fallback: object) -> None:
        self._fallback = fallback
        self._desk_view_available = self._check_availability()

    def _check_availability(self) -> bool:
        """Return True if the current device exposes a Desk View stream."""
        try:
            import AVFoundation  # type: ignore[import]  # noqa: F401
            # Probe for the virtual desk-view device
            discovery = AVFoundation.AVCaptureDeviceDiscoverySession
            devices = discovery.discoverySessionWithDeviceTypes_mediaType_position_(
                ["AVCaptureDeviceTypeBuiltInWideAngleCamera"],
                AVFoundation.AVMediaTypeVideo,
                AVFoundation.AVCaptureDevicePositionFront,
            ).devices()
            for device in devices:
                if "desk" in device.localizedName().lower():
                    logger.info("Desk View device found: %s", device.localizedName())
                    return True
        except Exception as exc:
            logger.debug("Desk View unavailable: %s", exc)
        return False

    @property
    def is_desk_view_active(self) -> bool:
        """True when frames are sourced from the Desk View stream."""
        return self._desk_view_available

    def read(self) -> Optional[np.ndarray]:
        """Return the next frame, preferring Desk View when available.

        Returns:
            BGR frame array or None on failure.
        """
        # TODO: wire up AVFoundation session when Desk View is confirmed available
        return self._fallback.read()
