# Validation

Live pose/clutch results remain unconfirmed. A synthetic test pass cannot establish usability over
a keyboard.

## 1. Finger counting

```sh
python scripts/mvp_pose_test.py --reps 3 --frames 30 --seed 42
```

This requests 360 scored frames across counts 1–4. Target: at least 95% overall accuracy over at
least 200 frames. Record per-class accuracy, camera placement, lighting, and dropped-hand
observations.

The script scores only frames with a detected hand. It can print PASS with fewer than 200 frames,
alongside a warning. Check the frame total yourself; a high score with frequent tracking loss is
insufficient. Curled hands, typing posture, transitions, and identity of the extended fingers need
separate coverage.

## 2. Typing lockout and arming

```sh
python scripts/mvp_clutch_test.py --mode typing --typing-sec 120
python scripts/mvp_clutch_test.py --mode latency --trials 12 --seed 42
```

Target: zero false gesture activations during the full typing session and median system arming
latency below 300 ms. Include typing pauses and ordinary hand repositioning. Confirm the keyboard
listener receives events.

System latency starts at the first valid smoothed pose, after camera capture and inference. It is
not end-to-end input latency. The report calculates its latency verdict from successful trials;
inspect timeouts and count mismatches separately. A run interrupted by camera loss also needs
repeating.

Save terminal output and a short setup description. Repeat with a changed hand position or lighting
before treating the setup as reliable.

## Automated tests

With the development dependencies installed:

```sh
python -m pytest -q
```

The existing suite covers homography, feature extraction, the older heuristic classifier/mode
manager, TCN behavior, and a synthetic pipeline. It does not cover the pose/clutch cursor pipeline.
Some mode-manager assertions are unconditional or conditional on a transition, so a pass can miss
broken behavior.

Next coverage should use synthetic landmarks and controlled timestamps for clutch timing, typing
override, tracking loss, pose changes, and exactly one click per pinch. An integration test should
assert emitted events through a fake output sink.

At the documentation review on 2026-11-10, `python3 -m pytest -q` could not run because pytest was
absent. No camera or OS-input trial was performed. The historical result of 29 passing tests has
not been reverified.
