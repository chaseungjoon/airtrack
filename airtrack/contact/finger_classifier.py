"""Passive friction acoustic classifier using spectral features + distance-weighted k-NN.

Physical basis: dragging N fingers across keyboard ridge gaps produces a friction
sound whose spectral complexity scales with contact points. More fingers → broader
spectrum, higher flatness, more complex waveform — independent of swipe force.

Feature vector (17 dims):
  [0]    spectral flatness  — Wiener entropy; ~1 = white noise (many contacts)
  [1]    zero-crossing rate — temporal complexity per sample
  [2]    spectral centroid  — centre-of-mass frequency (normalized 0–1 by Nyquist)
  [3]    spectral bandwidth — RMS spread around centroid (normalized)
  [4:17] MFCCs             — 13 mel-frequency cepstral coefficients (mean over frames)

Classifier: distance-weighted k-NN on z-score normalized feature space.
Training data persists in models/audio_features.npz across sessions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.signal import sosfilt
from scipy.fftpack import dct

from airtrack.shared.swipe_detector import load_wav, _make_hpf

logger = logging.getLogger(__name__)

Label = Literal["1 finger", "2 fingers", "3 fingers", "4 fingers"]

LABEL_TO_INT: dict[str, int] = {"1 finger": 1, "2 fingers": 2, "3 fingers": 3, "4 fingers": 4}
INT_TO_LABEL: dict[int, Label] = {1: "1 finger", 2: "2 fingers", 3: "3 fingers", 4: "4 fingers"}

_DEFAULT_FEATURES_PATH = "models/audio_features.npz"
_N_MFCC = 13
_N_MELS = 26
_N_FFT = 512
_FEATURE_DIM = 4 + _N_MFCC  # 17


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
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
    audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
    frame_len = min(int(0.025 * sr), len(audio))
    frame_step = int(0.010 * sr)
    window = np.hamming(frame_len)
    fbank = _mel_filterbank(sr, _N_FFT, _N_MELS)
    mfcc_frames = []
    for start in range(0, max(len(audio) - frame_len, 1), frame_step):
        frame = audio[start: start + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))
        mag = np.abs(np.fft.rfft(frame * window, n=_N_FFT))[: _N_FFT // 2 + 1]
        mel_e = np.maximum(fbank @ mag, 1e-10)
        coeffs = dct(np.log(mel_e), type=2, norm="ortho")[:n_mfcc]
        mfcc_frames.append(coeffs)
    return np.stack(mfcc_frames).mean(axis=0).astype(np.float32)


def extract_spectral_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract a 17-dim spectral feature vector from a swipe audio segment."""
    n = len(audio)
    nyquist = sr / 2.0
    seg = audio[:_N_FFT] if n >= _N_FFT else np.pad(audio, (0, _N_FFT - n))
    mag = np.abs(np.fft.rfft(seg)) + 1e-10
    freqs = np.fft.rfftfreq(_N_FFT, 1.0 / sr)

    spectral_flatness = float(np.exp(np.mean(np.log(mag))) / np.mean(mag))
    zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2.0)
    total_e = float(np.sum(mag))
    centroid_hz = float(np.sum(freqs * mag) / total_e) if total_e > 0 else 0.0
    centroid_norm = centroid_hz / nyquist
    bandwidth_hz = float(
        np.sqrt(np.sum((freqs - centroid_hz) ** 2 * mag) / total_e)
    ) if total_e > 0 else 0.0
    bandwidth_norm = bandwidth_hz / nyquist
    mfcc = extract_mfcc(audio, sr)
    return np.concatenate([[spectral_flatness, zcr, centroid_norm, bandwidth_norm], mfcc]).astype(np.float32)


class FingerClassifier:
    """Distance-weighted k-NN classifier on spectral features.

    Args:
        sample_rate: Audio sample rate in Hz.
        features_path: .npz file for persisting the training set.
        k: Number of neighbours (auto-capped to training set size).
    """

    def __init__(
        self,
        sample_rate: int,
        features_path: str = _DEFAULT_FEATURES_PATH,
        k: int = 5,
    ) -> None:
        self._sr = sample_rate
        self._features_path = Path(features_path)
        self._k = k
        self._sos = _make_hpf(sample_rate)

        self._X: np.ndarray = np.empty((0, _FEATURE_DIM), dtype=np.float32)
        self._y: np.ndarray = np.empty((0,), dtype=np.int32)
        self._mean: np.ndarray = np.zeros(_FEATURE_DIM, dtype=np.float32)
        self._std: np.ndarray = np.ones(_FEATURE_DIM, dtype=np.float32)

        self._load()

    @property
    def is_calibrated(self) -> bool:
        return len(set(self._y.tolist())) >= 2

    @property
    def n_samples(self) -> int:
        return int(len(self._y))

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        zi = np.zeros((self._sos.shape[0], 2))
        filtered, _ = sosfilt(self._sos, audio, zi=zi)
        return extract_spectral_features(filtered, self._sr)

    def classify_audio(self, audio: np.ndarray) -> Label:
        if not self.is_calibrated:
            return "3 fingers"
        return INT_TO_LABEL[self._knn(self.extract_features(audio))]

    def add_example_audio(self, audio: np.ndarray, finger_count: int) -> None:
        feat = self.extract_features(audio)
        self._X = np.vstack([self._X, feat[np.newaxis]])
        self._y = np.append(self._y, np.int32(finger_count))
        self._refit_normaliser()
        self._save()

    def fine_tune(self, wav_paths: list[str], true_finger_counts: list[int]) -> tuple[float, float]:
        n = min(len(wav_paths), len(true_finger_counts))
        if n == 0:
            return 0.0, 0.0
        new_feats: list[np.ndarray] = []
        for path in wav_paths[:n]:
            try:
                audio, _ = load_wav(path)
                new_feats.append(self.extract_features(audio))
            except Exception as exc:
                logger.warning("Could not load %s: %s", path, exc)
                new_feats.append(np.zeros(_FEATURE_DIM, dtype=np.float32))
        new_y = np.array(true_finger_counts[:n], dtype=np.int32)
        acc_before = self._accuracy_on(np.array(new_feats), new_y)
        self._X = np.vstack([self._X, np.array(new_feats)])
        self._y = np.concatenate([self._y, new_y])
        self._refit_normaliser()
        self._save()
        acc_after = self._loo_accuracy()
        return acc_before, acc_after

    def _save(self) -> None:
        self._features_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self._features_path, X=self._X, y=self._y, mean=self._mean, std=self._std)

    def _load(self) -> None:
        if not self._features_path.exists():
            return
        try:
            data = np.load(self._features_path)
            self._X = data["X"].astype(np.float32)
            self._y = data["y"].astype(np.int32)
            self._mean = data["mean"].astype(np.float32)
            self._std = data["std"].astype(np.float32)
        except Exception as exc:
            logger.warning("Could not load audio features: %s", exc)

    def _refit_normaliser(self) -> None:
        if len(self._X) == 0:
            return
        self._mean = self._X.mean(axis=0)
        self._std = self._X.std(axis=0) + 1e-8

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._std

    def _knn(self, feat: np.ndarray) -> int:
        X_norm = self._normalise(self._X)
        q_norm = self._normalise(feat)
        dists = np.sqrt(np.sum((X_norm - q_norm) ** 2, axis=1))
        k = min(self._k, len(self._y))
        nn_idx = np.argpartition(dists, k - 1)[:k]
        weights = 1.0 / (dists[nn_idx] + 1e-8)
        vote = np.zeros(5, dtype=np.float64)
        for lbl, w in zip(self._y[nn_idx], weights):
            vote[lbl] += w
        return int(np.argmax(vote[1:]) + 1)

    def _accuracy_on(self, X: np.ndarray, y: np.ndarray) -> float:
        if len(y) == 0 or not self.is_calibrated:
            return 0.0
        return sum(1 for feat, true in zip(X, y) if self._knn(feat) == true) / len(y)

    def _loo_accuracy(self) -> float:
        n = len(self._y)
        if n < 2:
            return 0.0
        X_norm = self._normalise(self._X)
        correct = 0
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_loo, y_loo = X_norm[mask], self._y[mask]
            dists = np.sqrt(np.sum((X_loo - X_norm[i]) ** 2, axis=1))
            k = min(self._k, len(y_loo))
            nn_idx = np.argpartition(dists, k - 1)[:k]
            weights = 1.0 / (dists[nn_idx] + 1e-8)
            vote = np.zeros(5, dtype=np.float64)
            for lbl, w in zip(y_loo[nn_idx], weights):
                vote[lbl] += w
            if int(np.argmax(vote[1:]) + 1) == self._y[i]:
                correct += 1
        return correct / n
