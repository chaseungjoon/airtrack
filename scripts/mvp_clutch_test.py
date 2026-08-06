#!/usr/bin/env python3
"""
AirTrack — Mode Discriminator Test  (Phase 2 gate)
===================================================
Validates Module 1 without audio (PLAN.md §2.1): keystroke lockout
(Modality A) + pose clutch (Modality B).

Two phases (both run by default):

  typing    Type normally in another window for --typing-sec seconds while
            the camera watches your hands. Every transition into GESTURE is
            a false activation. Gate: 0 false activations.

  latency   Prompted trials: hold up N fingers on cue. Measures
              - system latency: first valid pose frame → GESTURE armed
              - total latency: prompt shown → GESTURE armed (includes you)
            Gate: median system latency < 300 ms.

macOS permissions required: Camera + Input Monitoring (for pynput).

Usage:
    python scripts/mvp_clutch_test.py
    python scripts/mvp_clutch_test.py --mode typing --typing-sec 120
    python scripts/mvp_clutch_test.py --mode latency --trials 12
    python scripts/mvp_clutch_test.py --camera 1
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import cv2
except ImportError as exc:
    print(f"[ERROR] Missing dependency: {exc}")
    print("Run:  pip install opencv-python mediapipe numpy pynput")
    sys.exit(1)

from airtrack.action_state.keystroke import KeystrokeMonitor
from airtrack.action_state.pose_clutch import ClutchState, PoseClutch
from airtrack.contact.finger_pose import PoseFingerCounter
from airtrack.shared.camera_capture import CameraCapture
from airtrack.shared.hand_tracker import HandTracker

WINDOW_NAME = "AirTrack — Clutch Test"
LATENCY_GATE_MS = 300.0

_GREEN = (80, 220, 80)
_GRAY = (140, 140, 140)
_WHITE = (240, 240, 240)
_YELLOW = (60, 200, 255)
_RED = (60, 60, 255)

_STATE_COLORS = {
    ClutchState.TYPING: _GRAY,
    ClutchState.ARMING: _YELLOW,
    ClutchState.GESTURE: _GREEN,
}


def _header(msg: str) -> None:
    print(f"\n{'─' * 62}")
    print(f"  {msg}")
    print(f"{'─' * 62}")


def _draw_hud(frame, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    y = 34
    for text, color in lines:
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        y += 34


def _pump_window(frame) -> bool:
    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), 27)


class _Pipeline:
    """One frame step of the Module 1 stack: camera → pose → clutch."""

    def __init__(self, cam: CameraCapture, tracker: HandTracker, keys: KeystrokeMonitor) -> None:
        self._cam = cam
        self._tracker = tracker
        self._keys = keys
        self.counter = PoseFingerCounter()
        self.clutch = PoseClutch()

    def step(self):
        """Returns (frame, pose_count, state) or None when the camera ends."""
        frame = self._cam.read()
        if frame is None:
            return None
        hands = self._tracker.process(frame)
        hand = hands[0] if hands else None
        count = self.counter.update(hand)
        state = self.clutch.update(count, self._keys.seconds_since_last_press)
        return frame, count, state


# ── phase A: typing false activations ───────────────────────────────────────

def run_typing_phase(pipeline: _Pipeline, duration_sec: float) -> int | None:
    """Returns the false-activation count, or None if the user aborted."""
    _header(f"PHASE A — TYPING ({duration_sec:.0f}s)")
    print("  Focus another window (e.g. a text editor) and type normally.")
    print("  Keep your hands in camera view. Every GESTURE activation counts")
    print("  against the gate. q/Esc in the video window aborts.\n")
    input("  Press Enter to start... ")

    deadline = time.monotonic() + duration_sec
    false_activations = 0
    prev_state = ClutchState.TYPING

    while time.monotonic() < deadline:
        result = pipeline.step()
        if result is None:
            break
        frame, count, state = result
        if state is ClutchState.GESTURE and prev_state is not ClutchState.GESTURE:
            false_activations += 1
            print(f"  [!] false GESTURE activation #{false_activations} "
                  f"(pose count {pipeline.clutch.armed_count or count})")
        prev_state = state

        remaining = deadline - time.monotonic()
        _draw_hud(frame, [
            (f"TYPE NORMALLY — {remaining:.0f}s left", _WHITE),
            (f"state: {state.value}", _STATE_COLORS[state]),
            (f"false activations: {false_activations}",
             _RED if false_activations else _GREEN),
        ])
        if not _pump_window(frame):
            return None
    return false_activations


# ── phase B: arming latency ─────────────────────────────────────────────────

def _await_state(pipeline: _Pipeline, want: ClutchState, hud_text: str,
                 hold_sec: float = 1.0, timeout_sec: float = 15.0) -> bool:
    """Wait until the clutch has been in `want` continuously for hold_sec."""
    since: float | None = None
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        result = pipeline.step()
        if result is None:
            return False
        frame, _, state = result
        now = time.monotonic()
        if state is want:
            since = since or now
            if now - since >= hold_sec:
                return True
        else:
            since = None
        _draw_hud(frame, [(hud_text, _WHITE), (f"state: {state.value}", _STATE_COLORS[state])])
        if not _pump_window(frame):
            return False
    return True  # proceed anyway; the trial itself will catch real problems


def run_latency_phase(pipeline: _Pipeline, trials: int, seed: int | None) -> list[dict] | None:
    """Returns per-trial results, or None if the user aborted."""
    _header(f"PHASE B — ARMING LATENCY ({trials} trials)")
    print("  On each cue, hold up the prompted number of fingers until armed,")
    print("  then rest your hands in typing posture between trials.\n")
    input("  Press Enter to start... ")

    rng = random.Random(seed)
    results: list[dict] = []

    for i in range(1, trials + 1):
        if not _await_state(pipeline, ClutchState.TYPING,
                            "rest hands in typing posture..."):
            return None
        target = rng.randint(1, 4)
        pipeline.counter.reset()
        pipeline.clutch.reset()

        prompt_t = time.monotonic()
        first_pose_t: float | None = None
        armed_t: float | None = None
        deadline = prompt_t + 5.0

        while time.monotonic() < deadline:
            result = pipeline.step()
            if result is None:
                return None
            frame, count, state = result
            now = time.monotonic()
            if first_pose_t is None and count is not None and 1 <= count <= 4:
                first_pose_t = now
            if state is ClutchState.GESTURE:
                armed_t = now
            _draw_hud(frame, [
                (f"[{i}/{trials}] HOLD UP {target} FINGER{'S' if target > 1 else ''}", _YELLOW),
                (f"state: {state.value}", _STATE_COLORS[state]),
            ])
            if not _pump_window(frame):
                return None
            if armed_t is not None:
                break

        if armed_t is None or first_pose_t is None:
            print(f"  [{i}/{trials}] target {target}: TIMEOUT (no arm within 5s)")
            results.append({"target": target, "ok": False})
        else:
            system_ms = (armed_t - first_pose_t) * 1000
            total_ms = (armed_t - prompt_t) * 1000
            armed_count = pipeline.clutch.armed_count
            match = "✓" if armed_count == target else f"✗ armed as {armed_count}"
            print(f"  [{i}/{trials}] target {target}: system {system_ms:5.0f} ms | "
                  f"total {total_ms:5.0f} ms | count {match}")
            results.append({
                "target": target, "ok": True, "system_ms": system_ms,
                "total_ms": total_ms, "count_match": armed_count == target,
            })
    return results


# ── reporting ───────────────────────────────────────────────────────────────

def _print_verdict(false_activations: int | None, latency_results: list[dict] | None) -> None:
    _header("VERDICT")
    if false_activations is not None:
        ok = false_activations == 0
        print(f"  Typing gate: {false_activations} false activation(s) — "
              f"{'✅ PASS' if ok else '❌ FAIL (gate: 0)'}")
    if latency_results:
        armed = [r for r in latency_results if r["ok"]]
        print(f"  Latency trials armed: {len(armed)}/{len(latency_results)}")
        if armed:
            sys_lat = [r["system_ms"] for r in armed]
            med = statistics.median(sys_lat)
            correct = sum(1 for r in armed if r["count_match"])
            ok = med < LATENCY_GATE_MS
            print(f"  System arming latency: median {med:.0f} ms, max {max(sys_lat):.0f} ms — "
                  f"{'✅ PASS' if ok else f'❌ FAIL (gate: <{LATENCY_GATE_MS:.0f} ms)'}")
            print(f"  Armed with the prompted count: {correct}/{len(armed)}")
    print("\n  Tunables: CLUTCH_HOLD_MS / CLUTCH_RELEASE_MS / POSE_SMOOTHING_FRAMES")
    print("  in airtrack/config.py.")


# ── entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AirTrack mode discriminator test")
    parser.add_argument("--mode", choices=("all", "typing", "latency"), default="all")
    parser.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--typing-sec", type=float, default=60.0,
                        help="duration of the typing phase (default 60)")
    parser.add_argument("--trials", type=int, default=8,
                        help="latency trials (default 8)")
    parser.add_argument("--seed", type=int, default=None, help="trial prompt seed")
    args = parser.parse_args()

    false_activations: int | None = None
    latency_results: list[dict] | None = None

    with CameraCapture(index=args.camera) as cam, \
         HandTracker(max_hands=1) as tracker, \
         KeystrokeMonitor() as keys:
        pipeline = _Pipeline(cam, tracker, keys)
        if args.mode in ("all", "typing"):
            false_activations = run_typing_phase(pipeline, args.typing_sec)
            if false_activations is None:
                print("\n  Aborted by user.")
                cv2.destroyAllWindows()
                return
        if args.mode in ("all", "latency"):
            latency_results = run_latency_phase(pipeline, args.trials, args.seed)
            if latency_results is None:
                print("\n  Aborted by user.")

    cv2.destroyAllWindows()
    _print_verdict(false_activations, latency_results)


if __name__ == "__main__":
    main()
