# Ideas

## Goals

- Control the cursor freely while keeping hands at the keyboard
- No new hardware, trackpad input, assigned shortcut keys, or camera aimed at the keyboard.
- Existing Apple devices are an optional research branch; the MacBook-only branch must stand on its own.

These are experiments, not implemented features. Each must eventually support continuous motion,
stopping, clicking, and dragging. Recognizing a few swipes is insufficient.

## Directions

- [Software only](#1-software-only)
- [Hardware support](#2-hardware-support)

## 1. Software only

### Chassis vibration plus microphone sensing

Read the built-in inertial measurement unit (IMU) alongside microphone audio. Compare vibrations
through the chassis with sound from fingertip contact to estimate sliding direction and contact
changes. A learned motion model could combine these weak observations over time.

[Undocumented IMU access](https://github.com/olvvier/apple-silicon-accelerometer) exists on some
MacBooks. A [2026 preprint](https://arxiv.org/html/2609.21569v1) reports position-dependent keystroke
vibrations, not continuous sliding accuracy. Delivered sample rate and usable bandwidth need
measurement; native sensor specifications alone are insufficient.

**First test:** compare slow horizontal, vertical, and diagonal strokes with stationary contact,
typing, and desk movement. Determine whether motion remains distinguishable across pressure,
hand position, and supporting surfaces. Quiet sliding may produce too little usable information.

### Coded stereo acoustics

Emit distinguishable probes from the left and right speakers. Track phase, delay, and Doppler
changes in their reflections, then estimate relative motion from the calibrated geometry.
[Multiwave](https://crpittman.github.io/publication/multiwave-ea/) demonstrated multidimensional
gesture sensing with two speakers and a microphone; precise keyboard pointing remains unproven.

**First test:** measure each audio path and separate direct speaker leakage from changing echoes.
Check which microphone channels software actually exposes. Test continuous trajectories and
stops, including motion poorly observed by either speaker alone. Do not infer finger count from
volume or assume a high-frequency probe is inaudible to everyone.

### Active mechanical probing

**Speculative:** drive the built-in speakers with a known probe and measure the chassis response
through the IMU and microphones. Touching different keyboard regions may change mechanical
loading, damping, and resonance. Differential measurements could reveal contact even when a
stationary finger makes no friction sound.

**First test:** find a measurable response within the sensors' actual bandwidth, then compare
untouched, touched, and sliding conditions. Reject responses explained only by hand pressure or
desk support. Audible output and vibration must also be measured; this idea has no validated
MacBook pointing implementation.

### Screen-facing head and eye tracking

Keep the webcam in its ordinary position facing the user. Estimate head rotation and eye motion
for continuous pointing; investigate deliberate facial motion or contact signals for clicking.
Apple's [Head Pointer](https://support.apple.com/guide/mac-help/mchlb2d4782b/mac) establishes a
head-motion baseline. Accurate gaze, automatic intent detection, and comfortable dragging remain
separate research problems.

**First test:** distinguish intentional steering from reading, talking, and posture changes while
the hands remain on the keyboard. Eye tracking must preserve arbitrary cursor placement; snapping
between predefined UI targets cannot replace it.

### Learn from ordinary typing

Use key events as sparse labels for physical contact at known keyboard locations. Adapt audio and
vibration models to the user's hands, laptop, and desk. Which finger pressed the key remains a
latent variable; key events do not label the intervening trajectory.

Train intent detection on typing, pauses, repositioning, steering, and release. A pause alone must
never activate pointing. Keep ordinary keys as text input, and measure accidental key suppression
as well as false cursor movement. Validate motion on separately labeled sessions.

### Remaining hardware audit

| Component | Possible role or missing evidence |
|---|---|
| Keyboard controller / HID | Inspect exposed reports for useful timing or extra fields. No established stock-Mac path to continuous finger capacitance; key states alone cannot locate a hovering finger. |
| Lid angle / ambient light | Context for camera and acoustic calibration, not two-dimensional finger position. [Lid sensor access](https://github.com/samhenrigold/LidAngleSensor) varies by model. |
| Display / keyboard lighting | Controlled illumination may stabilize face images; it cannot reveal an occluded keyboard. |
| Wi-Fi / Bluetooth | Research firmware access to channel measurements before claiming RF hand sensing. Device proximity or signal strength alone is not a cursor trajectory. |
| Touch ID | The public [authentication interface](https://developer.apple.com/documentation/localauthentication) is not a raw fingerprint-motion stream. No established pointing path. |
| Fans / thermal / power telemetry | Characterize interference and operating state. No evidence here for recovering finger position from these signals. |
| Touch Bar on older models | [AppKit touch events](https://developer.apple.com/documentation/appkit/nstouchbar?changes=l_11) primarily supply horizontal motion. Could complement another signal, but cannot establish free two-dimensional pointing alone. |
| Trackpad | Excluded as an input surface. |

Start with synchronized chassis/audio measurements and stereo probes. Add a modality only after
it contributes information beyond the others. Compare held-out trajectories, resting drift,
end-to-end latency, false activations, and click/drag errors under normal typing posture.

## 2. Hardware support

### 2.1 First-party hardware: existing Apple products

No custom sensors, attachments, or required purchases. Explore devices the user already owns.

#### Apple Watch: wrist motion and local finger sensing

Use orientation changes for continuous steering while the hand stays near the home row. Test
relative-angle and angular-velocity mappings. Do not double-integrate acceleration and assume
drift-free finger position; finger motion can occur without corresponding wrist motion.

For finer control, investigate local acoustic sensing. [WatchHand](https://arxiv.org/html/2602.21610v1)
tracks hand joints with a watch's existing speaker and microphone. It reports 7.87 mm cross-session
mean joint error and about 115 ms processing per prediction on one device. It was evaluated on
Wear OS watches, not Apple Watch, and did not establish keyboard pointing performance.

**First test:** verify Apple Watch audio paths, wrist-down operation, and delivered motion latency.
Then estimate a small continuous steering space and contact/release state. Combine wrist and
MacBook vibrations to distinguish body movement from fingertip interaction. The
[released WatchHand dataset](https://github.com/witlab-kaist/WatchHand) provides a research starting
point, but not the ML implementation.

#### AirPods: head motion plus hand intent

[Core Motion](https://developer.apple.com/videos/play/wwdc2023/10179/) exposes supported headphones'
orientation, acceleration, and rotation rate directly on macOS 14 and later. Use head rotation for
cursor travel, with Watch or MacBook signals identifying hand intent and clicks. Also test coarse
head movement combined with fine wrist control.

**First test:** measure delivered rate, latency, neutral drift, and unintended movement during
reading and typing. Head orientation is not gaze. Do not assume separate synchronized motion
streams from both earbuds. Recenter during confidently identified typing without assigning a key.

#### iPhone: relay, reference, and another sensing point

Use the paired phone to relay Watch data to the Mac. If already resting on the same desk, its IMU
could help identify common desk vibration. This is a hypothesis; placement changes the signal.

UWB ranging may constrain wrist movement on supported devices.
[SmartPoser](https://chrisharrison.net/projects/smartposer/SmartPoser.pdf) combined iPhone/Watch
motion and UWB for arm estimation. That supports investigating drift correction, not assuming
finger-level accuracy or a complete two-dimensional position from a single distance measurement.

#### Active haptic probing

**Speculative:** emit a known Watch haptic pulse and inspect the motion response. Finger contact
or tendon tension might change its mechanical signature. First establish repeatable differences
between hand states. Available actuator control, sampling bandwidth, and perceptible disturbance
may prevent continuous sensing; no cursor-tracking result is established.

#### Apple API constraints to test

- [Double tap](https://developer.apple.com/documentation/watchos-apps/enabling-double-tap?changes=_2.)
  invokes an onscreen action, not a continuous finger-pose stream. Custom click/release sensing
  still needs investigation; do not depend on an always-available background gesture callback.
- Higher-rate [Watch motion](https://developer.apple.com/videos/play/wwdc2023/10179/) arrives in
  one-second batches. Use low-latency samples for interaction and measure their actual delivery.
- Optical pulse data is not an unrestricted real-time input channel. Its
  [SensorKit interface](https://developer.apple.com/documentation/sensorkit/srsensor/photoplethysmogram?changes=la_6)
  requires a research entitlement; Apple's internal gesture pipeline cannot be assumed available.
- Test [wrist-down execution](https://developer.apple.com/documentation/watchkit/using-extended-runtime-sessions?changes=_2__8&language=objc)
  and [network transport](https://developer.apple.com/documentation/technotes/tn3135-low-level-networking-on-watchos?changes=__7)
  on real devices. Public APIs, undocumented access, and research-only capabilities must be
  recorded separately. Availability varies by model and OS version.

### 2.2 Third-party hardware

Final resort