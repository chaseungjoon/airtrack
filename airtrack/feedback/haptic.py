"""macOS haptic feedback via NSHapticFeedbackManager (PyObjC).

AppKit is bundled inside pyobjc-framework-Cocoa — no separate package needed.
"""

from __future__ import annotations

import logging
from types import ModuleType
from typing import Any, Optional

logger = logging.getLogger(__name__)

_appkit: Optional[ModuleType] = None
try:
    import AppKit as _appkit  # type: ignore[import]
except ImportError:
    logger.warning("AppKit not available; haptic feedback disabled.")


class HapticFeedback:
    """Triggers MacBook Taptic Engine patterns via NSHapticFeedbackManager.

    Patterns: "generic", "alignment", "level_change".
    Silently no-ops on non-Apple hardware or without PyObjC.
    """

    _PATTERN_KEYS: dict[str, int] = {
        "generic": 0,       # NSHapticFeedbackPatternGeneric
        "alignment": 1,     # NSHapticFeedbackPatternAlignment
        "level_change": 2,  # NSHapticFeedbackPatternLevelChange
    }

    def __init__(self) -> None:
        self._performer: Any = None
        if _appkit is not None:
            try:
                self._performer = _appkit.NSHapticFeedbackManager.defaultPerformer()
            except Exception as exc:
                logger.warning("Could not acquire haptic performer: %s", exc)

    def trigger(self, pattern: str = "generic") -> None:
        """Fire a haptic pattern.

        Args:
            pattern: One of "generic", "alignment", "level_change".
        """
        if self._performer is None:
            return
        pattern_id = self._PATTERN_KEYS.get(pattern, 0)
        try:
            self._performer.performFeedbackPattern_performanceTime_(pattern_id, 0)
        except Exception as exc:
            logger.debug("Haptic trigger failed: %s", exc)
