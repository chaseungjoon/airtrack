# AirTrack
> **Your current MacBook keyboard is now a trackpad. Zero extra hardware, zero toggles.**

> **Seamless control powered by multimodal sensor fusion.**

*⚠️ **Note: AirTrack is currently a Work In Progress (WIP).** The vision-first pose pipeline is under active development.*

AirTrack is a software defined virtual trackpad for MacBooks that transforms the space over your keyboard into a gesture interface / trackpad. It is **vision-first**: MediaPipe hand landmarks from your webcam plus OS level keystroke dynamics — two deterministic, noise-immune signals — distinguish intentional typing from trackpad gestures. Finger count is read from hand *pose* (which fingers are extended), so no depth sensing, no microphones, and no contact detection are needed.

## 3 Module Architecture
1. **Action State Discriminator:** An OS keystroke lockout plus a **pose clutch** — gesture mode arms only when a deliberate extended-finger pose is held for ~120 ms — switches between `TYPING` and `GESTURE` modes without hotkeys.
2. **Kinematic Trajectory Decoder:** Uses 2D vision (front webcam or top-down Desk View) to translate Kalman-smoothed fingertip movement into OS level cursor commands, with thumb-index pinch as click/drag.
3. **Finger Pose Estimator:** Counts extended fingers from PIP joint angles, cross-checked by velocity-vector clustering during motion, to differentiate a 1-finger cursor move from a 2-finger scroll or a 3-finger swipe — no camera depth required.

> Earlier acoustic approaches (structure-borne swipe profiling, 20 kHz sonar) were validated and retired — ambient noise puts finger-friction sound below the mic's noise floor. See `PLAN.md` §0/§5 for the post-mortem.
