"""Heuristic-bootstrapped training data recorder for the TCN.

Usage:
    python -m airtrack.action_state.data_collector --output data/bootstrap.npz --duration 300
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from airtrack.shared.camera_capture import CameraCapture
from airtrack.shared.hand_tracker import HandTracker
from airtrack.action_state.bimanual import FeatureExtractor
from airtrack.action_state.discriminator import HybridStateClassifier, TrackMode
from airtrack.action_state.keystroke import KeystrokeMonitor

logger = logging.getLogger(__name__)

_LABEL_TYPING = 0
_LABEL_GESTURE = 1


class DataCollector:
    """Accumulates (feature_vector, label) pairs for TCN training."""

    def __init__(self, output_path: str, auto_label_only: bool = True) -> None:
        self._output_path = Path(output_path)
        self._auto_only = auto_label_only
        self._features: list[np.ndarray] = []
        self._labels: list[int] = []

    def record(self, feature: np.ndarray, label: int) -> None:
        self._features.append(feature.copy())
        self._labels.append(label)

    def save(self) -> None:
        if not self._features:
            logger.warning("No samples to save.")
            return
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self._output_path, X=np.stack(self._features), y=np.array(self._labels, dtype=np.int64))
        logger.info("Saved %d samples to %s", len(self._labels), self._output_path)

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

    collector.save()
    print(f"Collected {len(collector)} labelled samples.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="AirTrack training data collector")
    parser.add_argument("--output", default="data/bootstrap.npz")
    parser.add_argument("--duration", type=float, default=300.0)
    args = parser.parse_args()
    _collect(args.output, args.duration)
