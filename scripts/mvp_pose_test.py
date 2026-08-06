#!/usr/bin/env python3
"""
AirTrack — Vision Finger-Count Pose Test  (Phase 1 gate)
=========================================================
Validates Modality A of Module 3 (PLAN.md §2.3): counting extended fingers
from MediaPipe landmarks via PIP joint angles — no audio, no contact sensing.

Two modes:

  guided (default)  Prompts a shuffled sequence of finger counts (1–4).
                    For each prompt: countdown, then captures N frames and
                    scores the smoothed prediction against ground truth.
                    Ends with a per-class accuracy table and the Phase 1
                    kill-gate verdict (≥95% frame accuracy).

  --live            Free-run overlay only: shows the camera feed with
                    landmarks, per-finger extension state, smoothed count,
                    pinch state, and FPS. No scoring. Quit with q/Esc.

A preview window is shown in both modes — keep your whole hand in frame.
Works with the front webcam or a top-down Desk View / mirror rig.

Usage:
    python scripts/mvp_pose_test.py
    python scripts/mvp_pose_test.py --live
    python scripts/mvp_pose_test.py --reps 5 --frames 40
    python scripts/mvp_pose_test.py --camera 1
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import cv2
    import numpy as np
except ImportError as exc:
    print(f"[ERROR] Missing dependency: {exc}")
    print("Run:  pip install opencv-python mediapipe numpy")
    sys.exit(1)

from airtrack.contact.finger_pose import FingerPose, PoseFingerCounter
from airtrack.kinematic.pinch import PinchDetector
from airtrack.shared.camera_capture import CameraCapture
from airtrack.shared.hand_tracker import HandLandmarks, HandTracker

WINDOW_NAME = "AirTrack — Pose Test"
ACCURACY_GATE = 0.95

# MediaPipe hand skeleton edges for the overlay
_SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)
_FINGERTIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}

_GREEN = (80, 220, 80)
_GRAY = (140, 140, 140)
_WHITE = (240, 240, 240)
_YELLOW = (60, 200, 255)


# ── display helpers ─────────────────────────────────────────────────────────

def _header(msg: str) -> None:
    print(f"\n{'─' * 62}")
    print(f"  {msg}")
    print(f"{'─' * 62}")


def _draw_hand(frame: np.ndarray, hand: HandLandmarks, pose: FingerPose | None) -> None:
    h, w = frame.shape[:2]
    pts = (hand.landmarks[:, :2] * np.array([w, h])).astype(int)
    for a, b in _SKELETON_EDGES:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), _GRAY, 1, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame, (x, y), 3, _WHITE, -1, cv2.LINE_AA)
    if pose is None:
        return
    extended = {
        "thumb": pose.thumb, "index": pose.index, "middle": pose.middle,
        "ring": pose.ring, "pinky": pose.pinky,
    }
    for name, tip in _FINGERTIPS.items():
        color = _GREEN if extended[name] else _GRAY
        cv2.circle(frame, tuple(pts[tip]), 8, color, 2, cv2.LINE_AA)


def _draw_hud(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 34
    for text, color in lines:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        y += 34


def _pump_window(frame: np.ndarray) -> bool:
    """Show frame; return False if the user requested quit (q/Esc)."""
    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), 27)


# ── live mode ───────────────────────────────────────────────────────────────

def run_live(camera_index: int) -> None:
    _header("LIVE MODE — q/Esc to quit")
    counter = PoseFingerCounter()
    pinch = PinchDetector()
    fps_t0, fps_frames, fps = time.monotonic(), 0, 0.0

    with CameraCapture(index=camera_index) as cam, HandTracker(max_hands=1) as tracker:
        for frame in cam.stream():
            hands = tracker.process(frame)
            hand = hands[0] if hands else None
            count = counter.update(hand)
            clicking = pinch.update(hand)

            if hand is not None:
                _draw_hand(frame, hand, counter.last_pose)
            hud = [
                (f"count: {count if count is not None else '-'}",
                 _GREEN if count else _WHITE),
                (f"pinch: {'CLICK' if clicking else '-'}",
                 _YELLOW if clicking else _GRAY),
                (f"{fps:.0f} fps", _GRAY),
            ]
            _draw_hud(frame, hud)

            fps_frames += 1
            if time.monotonic() - fps_t0 >= 1.0:
                fps = fps_frames / (time.monotonic() - fps_t0)
                fps_t0, fps_frames = time.monotonic(), 0

            if not _pump_window(frame):
                break
    cv2.destroyAllWindows()


# ── guided evaluation ───────────────────────────────────────────────────────

def _generate_prompts(reps: int, seed: int | None) -> list[int]:
    prompts = [n for n in (1, 2, 3, 4) for _ in range(reps)]
    random.Random(seed).shuffle(prompts)
    return prompts


def _capture_prompt(
    cam: CameraCapture,
    tracker: HandTracker,
    counter: PoseFingerCounter,
    target: int,
    n_frames: int,
    settle_sec: float = 2.0,
) -> list[int] | None:
    """Countdown, then capture n_frames smoothed predictions with a hand visible.

    Returns per-frame predictions, or None if the user quit.
    """
    counter.reset()
    predictions: list[int] = []
    settle_deadline = time.monotonic() + settle_sec

    for frame in cam.stream():
        hands = tracker.process(frame)
        hand = hands[0] if hands else None
        count = counter.update(hand)
        if hand is not None:
            _draw_hand(frame, hand, counter.last_pose)

        settling = time.monotonic() < settle_deadline
        if settling:
            remaining = settle_deadline - time.monotonic()
            hud = [
                (f"HOLD UP {target} FINGER{'S' if target > 1 else ''} (thumb tucked)", _YELLOW),
                (f"capturing in {remaining:.1f}s", _WHITE),
                (f"live: {count if count is not None else '-'}", _GRAY),
            ]
        else:
            if hand is not None and count is not None:
                predictions.append(count)
            hud = [
                (f"HOLD UP {target} FINGER{'S' if target > 1 else ''}", _YELLOW),
                (f"capturing {len(predictions)}/{n_frames}", _GREEN),
                (f"live: {count if count is not None else 'no hand!'}",
                 _WHITE if hand is not None else (60, 60, 255)),
            ]
        _draw_hud(frame, hud)

        if not _pump_window(frame):
            return None
        if len(predictions) >= n_frames:
            return predictions
    return predictions or None


def _print_results(per_class: dict[int, list[int]]) -> None:
    _header("RESULTS")
    total_correct = 0
    total_frames = 0
    print(f"  {'target':>7} | {'frames':>6} | {'correct':>7} | {'acc':>6} | confusions")
    print(f"  {'-' * 7}-+-{'-' * 6}-+-{'-' * 7}-+-{'-' * 6}-+-{'-' * 20}")
    for target in sorted(per_class):
        preds = per_class[target]
        correct = sum(1 for p in preds if p == target)
        total_correct += correct
        total_frames += len(preds)
        wrong = [p for p in preds if p != target]
        confusion = ", ".join(
            f"{val}×{wrong.count(val)}" for val in sorted(set(wrong))
        ) or "—"
        acc = correct / len(preds) if preds else 0.0
        print(f"  {target:>7} | {len(preds):>6} | {correct:>7} | {acc:>5.1%} | {confusion}")

    overall = total_correct / total_frames if total_frames else 0.0
    print(f"\n  Overall: {total_correct}/{total_frames} frames = {overall:.1%}")
    if total_frames < 200:
        print(f"  [!] Only {total_frames} frames — gate requires ≥200. Rerun with more --reps/--frames.")
    if overall >= ACCURACY_GATE:
        print(f"  ✅ PASS — ≥{ACCURACY_GATE:.0%} gate met. Phase 1 modality confirmed.")
    else:
        print(f"  ❌ FAIL — below {ACCURACY_GATE:.0%} gate.")
        print("     Check camera framing first (whole hand visible, no clipping);")
        print("     then try adjusting POSE_EXTENSION_ANGLE_DEG in airtrack/config.py.")


def run_guided(camera_index: int, reps: int, n_frames: int, seed: int | None) -> None:
    prompts = _generate_prompts(reps, seed)
    _header(
        f"GUIDED EVAL — {len(prompts)} prompts × {n_frames} frames "
        f"({len(prompts) * n_frames} total, gate ≥{ACCURACY_GATE:.0%})"
    )
    print("  Hold up the prompted number of fingers (thumb tucked) and keep")
    print("  the whole hand in frame. q/Esc in the video window aborts.\n")

    counter = PoseFingerCounter()
    per_class: dict[int, list[int]] = {1: [], 2: [], 3: [], 4: []}

    with CameraCapture(index=camera_index) as cam, HandTracker(max_hands=1) as tracker:
        for i, target in enumerate(prompts, 1):
            print(f"  [{i}/{len(prompts)}] hold up {target} finger(s)...", flush=True)
            preds = _capture_prompt(cam, tracker, counter, target, n_frames)
            if preds is None:
                print("\n  Aborted by user.")
                break
            per_class[target].extend(preds)

    cv2.destroyAllWindows()
    if any(per_class.values()):
        _print_results(per_class)


# ── entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AirTrack vision finger-count pose test")
    parser.add_argument("--live", action="store_true", help="free-run overlay, no scoring")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--reps", type=int, default=3, help="prompts per finger count (default 3)")
    parser.add_argument("--frames", type=int, default=30, help="frames captured per prompt (default 30)")
    parser.add_argument("--seed", type=int, default=None, help="prompt shuffle seed")
    args = parser.parse_args()

    if args.live:
        run_live(args.camera)
    else:
        run_guided(args.camera, args.reps, args.frames, args.seed)


if __name__ == "__main__":
    main()
