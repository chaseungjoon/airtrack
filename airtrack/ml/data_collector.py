"""Records labelled training sequences for bootstrapping the TCN.

Usage:
    python -m airtrack.ml.data_collector --output data/bootstrap.npz --duration 300

Two label sources are supported:
  auto   — heuristic classifier labels clearly-typed / clearly-gestured windows
            (ambiguous windows are dropped)
  manual — user presses F1 to mark the last window as TYPING, F2 for GESTURE
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from airtrack.camera.capture import CameraCapture
from airtrack.vision.hand_tracker import HandTracker
from airtrack.ml.feature_extractor import FeatureExtractor
from airtrack.ml.state_classifier import HybridStateClassifier, TrackMode
from airtrack.input.keyboard_listener import KeystrokeMonitor

logger = logging.getLogger(__name__)

_LABEL_TYPING = 0
_LABEL_GESTURE = 1


class DataCollector:
    """Accumulates (feature_vector, label) pairs for TCN training.

    Args:
        output_path: Where to save the collected .npz dataset.
        auto_label_only: When True, only heuristically-certain windows are saved.
    """

    def __init__(self, output_path: str, auto_label_only: bool = True) -> None:
        self._output_path = Path(output_path)
        self._auto_only = auto_label_only
        self._features: list[np.ndarray] = []
        self._labels: list[int] = []

    def record(self, feature: np.ndarray, label: int) -> None:
        """Store one (feature, label) pair.

        Args:
            feature: Float32 feature vector from FeatureExtractor.
            label: 0 for TYPING, 1 for GESTURE.
        """
        self._features.append(feature.copy())
        self._labels.append(label)

    def save(self) -> None:
        """Flush all accumulated samples to the output file."""
        if not self._features:
            logger.warning("No samples to save.")
            return
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        X = np.stack(self._features)
        y = np.array(self._labels, dtype=np.int64)
        np.savez(self._output_path, X=X, y=y)
        logger.info("Saved %d samples to %s", len(y), self._output_path)

    def __len__(self) -> int:
        return len(self._labels)


def _collect(output: str, duration_sec: float) -> None:
    collector = DataCollector(output_path=output, auto_label_only=True)
    extractor = FeatureExtractor()
    classifier = HybridStateClassifier()
    deadline = time.monotonic() + duration_sec

    print(f"Recording for {duration_sec:.0f}s — type normally or make gestures.")
    print("Press Ctrl+C to stop early.")

    with CameraCapture() as cam, HandTracker() as tracker, KeystrokeMonitor() as keys:
        for frame in cam.stream():
            if time.monotonic() > deadline:
                break
            hands = tracker.process(frame)
            extractor.update(hands[0] if hands else None)
            features = extractor.extract()
            mode = classifier.classify(features, keys.keystroke_rate)

            if mode == TrackMode.TYPING:
                collector.record(features, _LABEL_TYPING)
            elif mode == TrackMode.GESTURE:
                collector.record(features, _LABEL_GESTURE)
            # Drop AMBIGUOUS

    collector.save()
    print(f"Collected {len(collector)} labelled samples.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="AirTrack training data collector")
    parser.add_argument("--output", default="data/bootstrap.npz")
    parser.add_argument("--duration", type=float, default=300.0)
    args = parser.parse_args()
    _collect(args.output, args.duration)
