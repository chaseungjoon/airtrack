# Next steps

AirTrack has enough components to attempt a small cursor prototype. Its viability over a laptop
keyboard is still unproven. The hardest question is whether the camera can see deliberate gestures
in a comfortable position while the clutch rejects typing, pauses, and casual hand movement.

## What the structure supports

The tracker, pose counter, clutch, motion parser, and output bridge are separate classes. This makes
it practical to test decisions without moving the real cursor. The MediaPipe wrapper already
isolates inference, so changing the tracking model should not require rewriting gesture logic.

The main obstacle is integration drift: the new interaction exists in demos, while the application
and older estimators still describe a different pipeline. More modalities would increase the number
of paths to reconcile.

## Development order

1. **Measure the interaction.** Run the [pose and clutch checks](validation.md) with hands near the
   keyboard. Compare this with holding a hand higher in view. Record tracking loss and discomfort as
   well as accuracy. If framing fails, address camera placement before tuning thresholds or training
   a model.
2. **Connect one cursor path.** Repair entry-point imports and packaging, then wire tracker → pose →
   clutch → motion → output. Start with cursor movement and one click per pinch. Add a dry-run
   output sink so the same pipeline can be inspected without OS events.
3. **Make interruptions predictable.** Stop output on typing or invalid tracking, reset motion on
   reacquisition, select a stable primary hand, and release mouse buttons on every exit path. Add
   tests for these cases before enabling drag.
4. **Tune mapping and calibration.** Compare relative cursor movement with the existing absolute
   homography approach. Use one coordinate convention, reject invalid corners, and save
   camera/resolution settings with calibration. Normalize pinch distance by hand size and expose
   sensitivity settings.
5. **Add two-finger scrolling.** Once move/click is usable, add scroll routing and test pose changes
   during motion. Keep desktop swipes and other gestures deferred until this smaller set works
   consistently.

## Decisions to defer

Keep the current package layout until the first pipeline works. Then isolate retired audio
experiments and make their dependencies optional; `contact/` can become a pose/gesture module at
that point.

Profile capture, inference, decision, and output time separately before adding multiprocessing. The
old targets of under 50 ms end-to-end latency and about ±2 px resting jitter remain goals.
`TARGET_FPS = 60` requests a capture rate; it is not a benchmark.

Keep the TCN optional until recorded failures show a gap in the pose/clutch rules. A later model
should be evaluated on independently labeled sessions; training on heuristic labels alone cannot
establish that it fixes those heuristics. Haptics, extra hardware, and app packaging can follow
evidence that the basic interaction is useful.
