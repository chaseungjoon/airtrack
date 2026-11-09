# Experiment history

## May 2026: initial pipeline and audio trials

The first prototype added camera tracking, four-point calibration, a heuristic typing/gesture
classifier, an optional TCN, cursor output, and synthetic tests. The reported result was 29
passing tests.

Audio trials attempted to infer finger count from swipe friction using calibrated RMS thresholds and
spectral features. Initial thresholds classified every swipe as the highest count; guided references
and detector resets addressed that calibration bug. Recordings and fitted parameters remain in
`data/swipes/` and `models/`.

Reported limitations included ambient noise masking swipe friction and ultrasonic trials limited
by hardware response, microphone processing, and direct speaker-to-mic leakage. These
observations motivated retiring audio as a required input. They are project findings, not a general
proof about every acoustic setup.

## August 2026: vision and pose clutch

The design changed from estimating surface contact to counting extended fingers.
`PoseFingerCounter`, `PoseClutch`, keystroke recency, and guided evaluation scripts were added. The
tracker migrated from the removed legacy MediaPipe Solutions API to the Tasks API, with automatic
model download.

Synthetic pose/clutch sanity checks and an unchanged 29-test pass were reported. Live pose
accuracy and typing false-activation measurements remain pending. The new components have not
been integrated into the main application.

## Retained experiments

| Area | Files |
|---|---|
| Acoustic typing and recording | `action_state/acoustic_profiler.py`, `shared/recorder.py`, `shared/swipe_detector.py` under `airtrack/` |
| Acoustic finger count and sonar | `airtrack/contact/finger_classifier.py`, `airtrack/contact/sonar.py` |
| Manual trials | `scripts/mvp_audio_test.py`, `scripts/mvp_audio_train.py`, `scripts/mvp_sonar_test.py` |

Audio is outside the existing vision prototype, though the older `ContactEstimator` still accepts
sonar. Current [research ideas](../IDEAS.md) revisit software-only sensing under the fixed hardware
constraints; earlier proposals for added sensors are outside that scope.
