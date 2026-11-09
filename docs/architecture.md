# Architecture

The existing prototype uses vision plus keyboard activity. The descriptions below document that
code, not a chosen input method for the finished project. Current constraints and sensing experiments
are in [IDEAS.md](../IDEAS.md); keyboard-facing camera placement is outside the project goal.

## Code layout

| Path | Responsibility |
|---|---|
| `airtrack/shared/` | Camera, MediaPipe tracker, calibration, haptics; also older audio utilities |
| `airtrack/action_state/` | Keyboard monitor, pose clutch, older heuristic/TCN classifier |
| `airtrack/contact/` | Pose counter and older contact-estimation experiments |
| `airtrack/kinematic/` | Motion smoothing, pinch detection, Quartz output |
| `airtrack/config.py` | Camera, model, pose, clutch, and calibration settings |
| `scripts/` | Interactive experiments and calibration |
| `tests/` | Synthetic unit and integration tests |
| `data/`, `models/` | Recorded audio, fitted audio parameters, and downloaded model |

The package boundaries are usable, but `contact/` still mixes the newer pose approach with the
retired contact approach. Audio also remains mixed into shared utilities and dependencies.

## Existing vision demos

```text
Camera → HandTracker → PoseFingerCounter → PoseClutch → state overlay
Keyboard monitor ────────────────────────────↑
```

`HandTracker` uses the MediaPipe Tasks API in VIDEO mode. `PoseFingerCounter` counts extended
fingers using a 140° joint-angle threshold and a five-frame majority vote. It excludes the thumb
from the count.

`PoseClutch` implements `TYPING → ARMING → GESTURE`. A stable count of 1–4 fingers arms it after 120
ms. Any key press within 150 ms forces `TYPING`; losing the pose for 200 ms releases it. These are
configurable defaults, not measured performance guarantees.

The clutch accepts any pose with the requested count range. It does not yet enforce specific
combinations such as index-only. Landmark noise, occlusion, and camera placement still need testing;
angle-based classification does not guarantee accuracy across views.

## Intended cursor pipeline

When the clutch is active, a gesture router should send one-finger motion to `KinematicDecoder`,
pinch transitions to mouse-button events, and two-finger motion to scrolling.

| Gesture | Current implementation |
|---|---|
| One finger: move cursor | Motion parser and Quartz bridge exist; disconnected from pose/clutch |
| Thumb-index pinch: click | Hysteresis detector exists; decoder currently clicks every active frame |
| Hold pinch: drag | No persistent mouse-down/drag/mouse-up output |
| Two fingers: scroll | Planned; no scroll output |
| Three fingers: desktop swipe | Planned; no swipe output |
| Four fingers: App Exposé | Reserved |

`main.py` still uses the old package paths and heuristic mode manager. `discriminator.py` does not
use `PoseClutch`. `ContactEstimator` still votes over bounding-box, velocity, and optional sonar
estimates rather than using the pose counter. Confidence-weighted pose fusion and multiprocessing
remain proposals.

Before connecting OS output, define resets for typing, hand loss, and hand changes. Reset motion and
pinch state and release any held mouse button. `KinematicDecoder.reset()` currently resets motion
and cursor position but leaves pinch state intact.
