# Getting started

Run commands from the repository root. The current entry points are the standalone demos;
`airtrack/main.py` still imports removed packages.

## Setup

The project declares Python 3.11 or newer. For the vision demos:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install opencv-python mediapipe numpy pynput
```

Allow Camera access for the terminal or IDE running Python. The clutch demo also needs Input
Monitoring for keyboard events. Neither demo moves the cursor.

The tracker downloads its model on first use to `models/hand_landmarker.task`. After that, inference
runs locally. Camera and model settings live in `airtrack/config.py`; both demos accept `--camera
N`.

## Demos

Start with the live overlay:

```sh
python scripts/mvp_pose_test.py --live
```

Keep the whole hand visible. Check finger count and pinch state, then follow the scored runs in
[Validation](validation.md). Press `q` or Escape in the preview window to quit.

## Calibration

```sh
python -m scripts.calibrate
```

Point at the keyboard corners in this order: top-left, top-right, bottom-right, bottom-left. Press
Space for each point, or `q` to quit. The script saves `models/calibration.npz` relative to the
working directory.

This is an experimental four-point homography. It records camera pixels; a caller must supply pixels
at the same resolution. The old main loop supplies normalized coordinates, so that path needs
correction before use. Calibration does not yet validate corner geometry or store camera profiles,
and is unnecessary for the pose/clutch demos.

## Development dependencies

`requirements.txt` includes the broader test, ML, and audio dependencies. The lists in it and
`pyproject.toml` differ, and the configured `setuptools.backends.legacy:build` backend needs
correction before using editable installation. Use the direct demo setup above for now.
