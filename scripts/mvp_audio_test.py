#!/usr/bin/env python3
"""
AirTrack — Acoustic Finger-Count Test
=======================================
Pure evaluation: prompts N free-choice swipes (default 10), then asks
you to label what you swiped. Computes and displays classification accuracy.

No training or calibration here — run mvp_audio_train.py first.

Usage:
    python scripts/mvp_audio_test.py
    python scripts/mvp_audio_test.py --n-swipes 15
    python scripts/mvp_audio_test.py --device N
    python scripts/mvp_audio_test.py --list-devices
"""

from __future__ import annotations

import argparse
import sys
import time
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

from airtrack.audio.recorder import MicRecorder
from airtrack.audio.swipe_detector import SwipeDetector, SwipeEvent
from airtrack.audio.finger_classifier import FingerClassifier, LABEL_TO_INT


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


# ── ground truth input ──────────────────────────────────────────────────────

def _parse_ground_truth(raw: str, expected_count: int) -> list[int] | None:
    parts = raw.strip().split()
    if len(parts) != expected_count:
        print(f"  [!] Enter exactly {expected_count} values. Got {len(parts)}.")
        return None
    result: list[int] = []
    for p in parts:
        try:
            n = int(p)
            if n not in (1, 2, 3, 4):
                raise ValueError
            result.append(n)
        except ValueError:
            print(f"  [!] '{p}' is not valid — use only 1, 2, 3, or 4.")
            return None
    return result


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AirTrack — acoustic test: evaluate classifier accuracy"
    )
    parser.add_argument("--device", type=int, default=None,
                        help="Microphone device index (see --list-devices)")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--save-dir", type=str,
                        default=f"data/swipes/test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help="Directory for WAV files")
    parser.add_argument("--n-swipes", type=int, default=10,
                        help="Number of evaluation swipes (default 10)")
    args = parser.parse_args()

    if args.list_devices:
        print("\nAvailable audio INPUT devices:")
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:  # type: ignore[index]
                print(f"  [{i:2d}]  {dev['name']}")  # type: ignore[index]
        return

    _header("AirTrack — Acoustic Finger-Count Test")
    print(
        f"  {args.n_swipes} swipes — choose any finger count for each.\n"
        "  After all swipes, you'll enter what you actually did.\n"
        "  No training happens here — run mvp_audio_train.py to build data.\n"
    )

    # ── check training data ───────────────────────────────────────────────────
    classifier = FingerClassifier(sample_rate=44100)  # temp instance to check
    if not classifier.is_calibrated:
        print(
            "  [!] No training data found.\n"
            "  Run: python scripts/mvp_audio_train.py\n"
        )
        sys.exit(1)

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
    # Re-instantiate with the correct sample rate from the recorder
    classifier = FingerClassifier(sample_rate=recorder.sample_rate)
    scale = detector.onset_threshold * 4

    print(f"\n  Onset threshold : {detector.onset_threshold:.5f}")
    print(f"  Training set    : {classifier.n_samples} sample(s)")

    # ── swiping phase ─────────────────────────────────────────────────────────
    _header(f"Swiping  (0/{args.n_swipes})")
    print(
        "  Swipe with any number of fingers — 1, 2, 3, or 4.\n"
        "  You'll label them all at the end.\n"
        "  Press Enter to start...",
        end="",
    )
    try:
        input()
    except KeyboardInterrupt:
        recorder.stop()
        return

    print()
    n_swipes = args.n_swipes
    predictions: list[str] = []
    events: list[SwipeEvent] = []

    try:
        for i in range(1, n_swipes + 1):
            prefix = f"[{i:>2d}/{n_swipes}]  swipe #{i}"
            event = _collect_until_detected(recorder, detector, prefix)

            pred = classifier.classify_audio(event.audio)
            predictions.append(pred)
            events.append(event)

            bar = _bar(event.peak_rms, scale)
            print(
                f"\r  [{i:>2d}/{n_swipes}]  detected  →  {pred:<10}"
                f"  rms={event.peak_rms:.5f}  |{bar}|" + " " * 5
            )
            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\n\n  Interrupted early.")

    recorder.stop()

    if not predictions:
        print("\n  No swipes recorded. Exiting.")
        return

    actual_n = len(predictions)

    # ── ground truth input ────────────────────────────────────────────────────
    _header("Ground truth input")
    pred_display = " ".join(
        str(LABEL_TO_INT[p]) for p in predictions
    )
    print(f"  {actual_n} swipe(s) recorded.")
    print(f"  My predictions (finger counts):  {pred_display}\n")
    print(
        f"  Enter the actual finger counts in order (space-separated, 1–4)\n"
        f"  e.g.  1 2 3 2 4 1 2 3 2 4 :"
    )

    truth: list[int] | None = None
    while truth is None:
        raw = input("  > ")
        truth = _parse_ground_truth(raw, actual_n)

    # ── results ───────────────────────────────────────────────────────────────
    _header("Results")
    print(f"\n  {'#':>3}  {'Predicted':<12}  {'Truth':>5}  Match")
    print(f"  {'─'*3}  {'─'*12}  {'─'*5}  {'─'*5}")

    n_correct = 0
    for i, (pred, gt) in enumerate(zip(predictions, truth)):
        pred_int = LABEL_TO_INT[pred]
        ok = "✓" if pred_int == gt else "✗"
        if pred_int == gt:
            n_correct += 1
        print(f"  {i+1:>3}  {pred:<12}  {gt:>5}  {ok}")

    acc = n_correct / actual_n if actual_n else 0.0
    print(f"\n  Accuracy : {n_correct}/{actual_n}  ({acc * 100:.1f}%)\n")

    if acc >= 0.80:
        print("  ✓  Solid — §3.2 acoustic discrimination is viable!")
    elif acc >= 0.60:
        print("  ~  Moderate. Run more training sessions to improve.")
    else:
        print(
            "  ✗  Low accuracy. Tips:\n"
            "     • Run more  mvp_audio_train.py  sessions\n"
            "     • Try to swipe with a noticeably different spread per finger count\n"
            "     • Use --recalibrate in train to start fresh if data is noisy"
        )

    _header("Done")
    print(f"  WAV files : {args.save_dir}/\n")


if __name__ == "__main__":
    main()
