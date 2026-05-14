"""Modality B — Structure-borne acoustic profiling for TYPING vs GESTURE.

Typing generates high-amplitude, short-duration impact transients (key strikes).
Gestures (swipes) generate continuous, low-amplitude friction signals.
This module discriminates between the two using MFCC features from the
built-in microphone, exploiting the structural vibrations of the chassis.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfilt
from scipy.fftpack import dct

from airtrack.shared.recorder import MicRecorder

_HPF_CUTOFF_HZ = 150
_HPF_ORDER = 2
_N_MFCC = 13
_N_FFT = 512
_WINDOW_SEC = 0.1          # 100 ms rolling window for real-time classification
_IMPACT_RMS_MULTIPLIER = 8  # noise_floor × N → key-strike onset threshold


def _make_hpf(sr: int) -> np.ndarray:
    nyq = sr / 2.0
    return butter(_HPF_ORDER, _HPF_CUTOFF_HZ / nyq, btype="high", output="sos")  # type: ignore[return-value]


def _mel_filterbank(sr: int, n_fft: int, n_mels: int = 26) -> np.ndarray:
    low_mel = 0.0
    high_mel = 2595.0 * np.log10(1.0 + (sr / 2.0) / 700.0)
    mel_pts = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    fbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_l, f_c, f_r = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(f_l, f_c):
            if f_c > f_l:
                fbank[m - 1, k] = (k - f_l) / (f_c - f_l)
        for k in range(f_c, f_r):
            if f_r > f_c:
                fbank[m - 1, k] = (f_r - k) / (f_r - f_c)
    return fbank


def extract_mfcc(audio: np.ndarray, sr: int, n_mfcc: int = _N_MFCC) -> np.ndarray:
    """Compute mean MFCC over the audio segment."""
    pre = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    frame_len = min(int(0.025 * sr), len(pre))
    frame_step = int(0.010 * sr)
    fbank = _mel_filterbank(sr, _N_FFT)
    window = np.hamming(frame_len)
    frames = []
    for start in range(0, max(len(pre) - frame_len, 1), frame_step):
        f = pre[start: start + frame_len]
        if len(f) < frame_len:
            f = np.pad(f, (0, frame_len - len(f)))
        mag = np.abs(np.fft.rfft(f * window, n=_N_FFT))[: _N_FFT // 2 + 1]
        mel_e = np.maximum(fbank @ mag, 1e-10)
        coeffs = dct(np.log(mel_e), type=2, norm="ortho")[:n_mfcc]
        frames.append(coeffs)
    if not frames:
        return np.zeros(n_mfcc, dtype=np.float32)
    return np.stack(frames).mean(axis=0).astype(np.float32)


class AcousticProfiler:
    """Continuous mic monitor that classifies sound as IMPACT (typing) or FRICTION (gesture).

    Runs in a background thread alongside MicRecorder.

    Args:
        recorder: Running MicRecorder instance.
        noise_floor_rms: Ambient noise level (from recorder.measure_noise_floor).
    """

    IMPACT = "impact"    # key strike — suppresses gesture pipeline
    FRICTION = "friction"  # swipe — enables gesture pipeline
    SILENT = "silent"

    def __init__(self, recorder: MicRecorder, noise_floor_rms: float) -> None:
        self._recorder = recorder
        self._impact_thr = noise_floor_rms * _IMPACT_RMS_MULTIPLIER
        self._sr = recorder.sample_rate
        self._sos = _make_hpf(self._sr)
        self._zi = np.zeros((self._sos.shape[0], 2))

        self._window: deque[np.ndarray] = deque(
            maxlen=max(1, int(_WINDOW_SEC * self._sr / 512))
        )
        self._current: str = self.SILENT
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def current_class(self) -> str:
        """Latest acoustic classification: 'impact', 'friction', or 'silent'."""
        with self._lock:
            return self._current

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _run(self) -> None:
        while self._running:
            chunk = self._recorder.read()
            if chunk is None:
                time.sleep(0.002)
                continue

            filtered, self._zi = sosfilt(self._sos, chunk, zi=self._zi)
            self._window.append(filtered)

            if len(self._window) < (self._window.maxlen or 1):
                continue

            audio = np.concatenate(list(self._window))
            rms = float(np.sqrt(np.mean(audio ** 2)))

            if rms < self._impact_thr * 0.3:
                label = self.SILENT
            elif rms > self._impact_thr:
                label = self.IMPACT
            else:
                label = self.FRICTION

            with self._lock:
                self._current = label
