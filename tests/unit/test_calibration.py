"""Unit tests for HomographyCalibration."""

import numpy as np
import pytest

from airtrack.vision.calibration import HomographyCalibration, CalibrationError


def _make_calibrated() -> HomographyCalibration:
    cal = HomographyCalibration(save_path="/tmp/test_cal.npz")
    # Four corners of a 640x480 region
    for x, y in [(0, 0), (640, 0), (640, 480), (0, 480)]:
        cal.add_point(x, y)
    return cal


def test_requires_four_points() -> None:
    cal = HomographyCalibration(save_path="/tmp/test_cal.npz")
    assert not cal.is_calibrated
    cal.add_point(0, 0)
    assert not cal.is_calibrated


def test_calibrated_after_four_points() -> None:
    cal = _make_calibrated()
    assert cal.is_calibrated


def test_transform_corners() -> None:
    cal = _make_calibrated()
    kx, ky = cal.transform(0, 0)
    assert abs(kx) < 0.05 and abs(ky) < 0.05

    kx, ky = cal.transform(640, 480)
    assert abs(kx - 1.0) < 0.05 and abs(ky - 1.0) < 0.05


def test_transform_raises_before_calibration() -> None:
    cal = HomographyCalibration(save_path="/tmp/test_cal.npz")
    with pytest.raises(CalibrationError):
        cal.transform(100, 100)


def test_reset_clears_state() -> None:
    cal = _make_calibrated()
    cal.reset()
    assert not cal.is_calibrated


def test_save_and_load(tmp_path: object) -> None:
    save_file = str(tmp_path) + "/cal.npz"  # type: ignore[operator]
    cal = HomographyCalibration(save_path=save_file)
    for x, y in [(0, 0), (640, 0), (640, 480), (0, 480)]:
        cal.add_point(x, y)
    cal.save()

    cal2 = HomographyCalibration(save_path=save_file)
    cal2.load()
    assert cal2.is_calibrated
    kx, ky = cal2.transform(320, 240)
    assert 0.3 < kx < 0.7 and 0.3 < ky < 0.7
