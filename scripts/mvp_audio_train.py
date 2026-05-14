#!/usr/bin/env python3
"""
AirTrack — Acoustic Training
=============================
Collects labelled swipe examples to build the spectral k-NN classifier.
The script tells you which finger count to swipe each time; detection is
automatic via a 3-second window — no keypresses between swipes.

Run this before mvp_audio_test.py. Re-run as many times as needed;
each session adds more examples and improves accuracy.

Usage:
    python scripts/mvp_audio_train.py
    python scripts/mvp_audio_train.py --min-reps 5 --max-reps 10
    python scripts/mvp_audio_train.py --fingers 1 2 3 4
    python scripts/mvp_audio_train.py --seed 42
    python scripts/mvp_audio_train.py --recalibrate    # wipe saved data
    python scripts/mvp_audio_train.py --list-devices
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import sounddevice as sd
except ImportError as exc:
    print(f"[ERROR] Missing dependency: {exc}")
    print("Run:  pip install sounddevice scipy numpy")
    sys.exit(1)

from airtrack.shared.recorder import MicRecorder
from airtrack.shared.swipe_detector import SwipeDetector, SwipeEvent
from airtrack.contact.finger_classifier import FingerClassifier, LABEL_TO_INT, INT_TO_LABEL


# ── display helpers ─────────────────────────────────────────────────────────

def _bar(value: float, max_val: float, width: int = 16) -> str:
    filled = int(round(min(value / max(max_val, 1e-9), 1.0) * width))
    return "█" * filled + "░" * (width - filled)


def _header(msg: str) -> None:
    print(f"\n{'─' * 62}")
    print(f"  {msg}")
    print(f"{'─' * 62}")


# ── swipe collection ────────────────────────────────────────────────────────

def _await_swipe(
    recorder: MicRecorder,
    detector: SwipeDetector,
    prefix: str,
    timeout_sec: float = 5.0,
) -> "SwipeEvent | None":
    """Listen for up to timeout_sec, showing a live countdown. Returns None on timeout."""
    deadline = time.monotonic() + timeout_sec
    last_display = 0.0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            detector.reset()
            return None
        now = time.monotonic()
        if now - last_display >= 0.1:
            sys.stdout.write(f"\r  {prefix}  [{remaining:.1f}s]   ")
            sys.stdout.flush()
            last_display = now
        chunk = recorder.read()
        if chunk is None:
            time.sleep(0.002)
            continue
        event = detector.process_chunk(chunk)
        if event is not None:
            detector.reset()
            return event


def _collect_until_detected(
    recorder: MicRecorder,
    detector: SwipeDetector,
    prefix: str,
    timeout_sec: float = 5.0,
) -> SwipeEvent:
    """Repeat 3-second window until a swipe is detected; always returns an event."""
    attempt = 0
    while True:
        if attempt > 0:
            print(f"\r  {prefix}  [no swipe — try again]" + " " * 15)
        event = _await_swipe(recorder, detector, prefix, timeout_sec)
        if event is not None:
            return event
        attempt += 1


# ── session helpers ─────────────────────────────────────────────────────────

def _generate_prompts(
    fingers: list[int],
    min_reps: int,
    max_reps: int,
    seed: int | None,
) -> list[int]:
    """Return a randomised flat list of finger-count prompts."""
    rng = random.Random(seed)
    prompts: list[int] = []
    for n in fingers:
        prompts.extend([n] * rng.randint(min_reps, max_reps))
    rng.shuffle(prompts)
    return prompts


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AirTrack — acoustic training: collect labelled swipe data"
    )
    parser.add_argument("--device", type=int, default=None,
                        help="Microphone device index (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--save-dir", type=str,
                        default=f"data/swipes/train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help="Directory for WAV files")
    parser.add_argument("--min-reps", type=int, default=5,
                        help="Min repetitions per finger count (default 5)")
    parser.add_argument("--max-reps", type=int, default=10,
                        help="Max repetitions per finger count (default 10)")
    parser.add_argument("--fingers", type=int, nargs="+", default=[1, 2, 3, 4],
                        metavar="N", help="Which finger counts to collect (default: 1 2 3 4)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible prompt order")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Wipe all saved training data and start fresh")
    args = parser.parse_args()

    if args.list_devices:
        print("\nAvailable audio INPUT devices:")
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:  # type: ignore[index]
                print(f"  [{i:2d}]  {dev['name']}")  # type: ignore[index]
        return

    _header("AirTrack — Acoustic Training")
    print(
        "  Collects labelled swipe examples for the spectral k-NN classifier.\n"
        "  Each prompt gives a 3-second window — no keypresses between swipes.\n"
        "  Re-run to keep adding data; more examples = better accuracy.\n"
    )

    if args.recalibrate:
        p = Path("models/audio_features.npz")
        if p.exists():
            p.unlink()
            print("  Deleted saved training data — starting fresh.\n")

    # ── open mic ──────────────────────────────────────────────────────────────
    recorder = MicRecorder(device=args.device)
    try:
        recorder.start()
    except Exception as exc:
        print(f"\n[ERROR] Could not open microphone: {exc}")
        print("  Use --list-devices to pick a valid device index.")
        sys.exit(1)

    print("\n  Measuring noise floor — stay quiet for 2 seconds...")
    noise_floor = recorder.measure_noise_floor(duration_sec=2.0)
    print(f"  Noise floor RMS : {noise_floor:.5f}")
    if noise_floor > 0.05:
        print("  [!] High ambient noise — consider a quieter environment.")

    detector = SwipeDetector(
        sample_rate=recorder.sample_rate,
        noise_floor_rms=noise_floor,
        save_dir=args.save_dir,
    )
    classifier = FingerClassifier(sample_rate=recorder.sample_rate)
    scale = detector.onset_threshold * 4

    print(f"\n  Onset threshold : {detector.onset_threshold:.5f}")
    if classifier.n_samples > 0:
        print(f"  Training set    : {classifier.n_samples} sample(s) already stored")
    else:
        print("  Training set    : empty — this session starts it")

    # ── session plan ──────────────────────────────────────────────────────────
    valid_fingers = sorted(set(args.fingers) & {1, 2, 3, 4})
    if not valid_fingers:
        print("[ERROR] --fingers must contain values from {1, 2, 3, 4}.")
        recorder.stop()
        sys.exit(1)

    prompts = _generate_prompts(valid_fingers, args.min_reps, args.max_reps, args.seed)
    total = len(prompts)
    counts = Counter(prompts)

    _header("Session plan")
    print(f"  {total} total swipes  (randomised order)\n")
    for n in sorted(counts):
        print(f"  {n} finger(s)  [{INT_TO_LABEL[n]:<10}]  ×{counts[n]}")
    print(f"\n  Press Enter to begin...", end="")
    try:
        input()
    except KeyboardInterrupt:
        recorder.stop()
        return

    # ── collection loop ───────────────────────────────────────────────────────
    _header("Collection")
    print()

    n_collected = 0
    try:
        for i, finger_count in enumerate(prompts, start=1):
            label_str = INT_TO_LABEL[finger_count]
            prefix = f"[{i:>3d}/{total}]  {finger_count}f [{label_str}]  swipe now"

            event = _collect_until_detected(recorder, detector, prefix)
            n_collected += 1

            # Classify with current model and print result on same line
            if classifier.is_calibrated:
                pred = classifier.classify_audio(event.audio)
                mark = "✓" if LABEL_TO_INT[pred] == finger_count else "✗"
                bar = _bar(event.peak_rms, scale)
                print(
                    f"\r  [{i:>3d}/{total}]  {mark}  {finger_count}f → {pred:<10}"
                    f"  |{bar}|  {Path(event.wav_path).name}" + " " * 5
                )
            else:
                print(
                    f"\r  [{i:>3d}/{total}]  ✓  {finger_count}f detected"
                    f"  rms={event.peak_rms:.5f}" + " " * 20
                )

            # Add labelled example to training set immediately
            classifier.add_example_audio(event.audio, finger_count)
            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\n\n  Interrupted — saving data collected so far.")

    recorder.stop()

    # ── summary ───────────────────────────────────────────────────────────────
    _header("Training complete")
    print(f"  Collected this session : {n_collected} swipe(s)")
    print(f"  Training set total     : {classifier.n_samples} sample(s)")
    print(f"\n  WAV files  : {args.save_dir}/")
    print(  "  Model data : models/audio_features.npz")
    print(  "\n  Run  python scripts/mvp_audio_test.py  to evaluate.\n")


if __name__ == "__main__":
    main()
