# Next steps

The research goal is continuous cursor control with hands at the keyboard, no trackpad, no assigned
shortcut keys, and no new hardware. [IDEAS.md](../IDEAS.md) separates MacBook-only experiments from
optional use of existing Apple devices.

## Experiment order

1. **Inventory accessible signals.** Check the actual MacBook's IMU, audio channels, sample rates,
   timestamps, and processing. Record unavailable or undocumented paths. Do the same for any
   existing Watch or AirPods used in a separate experiment.
2. **Measure continuous motion.** Start with synchronized chassis/audio sensing and stereo acoustic
   probes. Compare trajectories, stops, and contact changes with typing and repositioning. Test on
   another session and desk surface before trusting a fitted model.
3. **Test intent and button state.** Measure false cursor activation during typing pauses, accidental
   text suppression, clicks, held drags, and release. No shortcut key may hide a detection failure.
4. **Connect a proven signal.** Repair entry-point imports and packaging, then use a fake output sink
   before enabling Quartz events. Reset movement and release buttons on missing data or typing.
5. **Evaluate the whole interaction.** Measure latency, drift, precision, typing-to-pointing transition,
   and comfort. Gesture-classification accuracy alone cannot establish cursor usability.

## Existing code

Camera tracking, pose/clutch demos, motion smoothing, and output classes are reusable experiments.
They do not establish the new sensing method. Main imports are stale, calibration units disagree,
and held pinches repeatedly click. See [architecture](architecture.md) for the implementation map
and [validation](validation.md) for the older tests.
