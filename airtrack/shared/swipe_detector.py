"""Onset/offset swipe detector with adaptive noise threshold and WAV saving.

State machine per chunk:
  IDLE  ─(rms > onset_thr)──►  COLLECTING
  COLLECTING  ─(rms < offset_thr, N consecutive)──►  FINALIZING ──► IDLE
  COLLECTING  ─(duration > max_dur)──► FINALIZING ──► IDLE

A 2-chunk (~23 ms) pre-roll is prepended to every detected swipe so the
full transient onset is captured.
"""

from __future__ import annotations

import time
import wave
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfilt


_HPF_CUTOFF_HZ: int = 150
_HPF_ORDER: int = 2
_PRE_ROLL_CHUNKS: int = 2
_OFFSET_CONFIRM_CHUNKS: int = 4
_MIN_DURATION_SEC: float = 0.06
_MAX_DURATION_SEC: float = 2.5


class _State(Enum):
    IDLE = auto()
    COLLECTING = auto()
    OFFSET_WAIT = auto()


@dataclass
class SwipeEvent:
    """One detected keyboard swipe with its audio and metadata."""
    audio: np.ndarray
    duration_sec: float
    peak_rms: float
    timestamp: float
    wav_path: str = ""


def _make_hpf(sample_rate: int) -> np.ndarray:
    nyq = sample_rate / 2.0
    return butter(_HPF_ORDER, _HPF_CUTOFF_HZ / nyq, btype="high", output="sos")  # type: ignore[return-value]


def save_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    int_audio = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_audio.tobytes())


def load_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "r") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    return audio, sr


class SwipeDetector:
    """Processes audio chunks in real-time and emits SwipeEvent objects."""

    def __init__(
        self,
        sample_rate: int,
        noise_floor_rms: float,
        onset_multiplier: float = 5.0,
        offset_multiplier: float = 2.5,
        save_dir: str = "data/swipes",
    ) -> None:
        self._sr = sample_rate
        self._onset_thr = noise_floor_rms * onset_multiplier
        self._offset_thr = noise_floor_rms * offset_multiplier
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)

        self._sos = _make_hpf(sample_rate)
        self._zi = np.zeros((self._sos.shape[0], 2))

        self._state = _State.IDLE
        self._buffer: list[np.ndarray] = []
        self._pre_roll: deque[np.ndarray] = deque(maxlen=_PRE_ROLL_CHUNKS)
        self._offset_count: int = 0
        self._swipe_start: float = 0.0
        self._swipe_index: int = 0

    @property
    def onset_threshold(self) -> float:
        return self._onset_thr

    @property
    def offset_threshold(self) -> float:
        return self._offset_thr

    def process_chunk(self, chunk: np.ndarray) -> Optional[SwipeEvent]:
        filtered, self._zi = sosfilt(self._sos, chunk, zi=self._zi)
        rms = float(np.sqrt(np.mean(filtered ** 2)))

        if self._state == _State.IDLE:
            self._pre_roll.append(filtered.copy())
            if rms > self._onset_thr:
                self._state = _State.COLLECTING
                self._buffer = list(self._pre_roll)
                self._swipe_start = time.monotonic()
                self._offset_count = 0

        elif self._state == _State.COLLECTING:
            self._buffer.append(filtered.copy())
            elapsed = time.monotonic() - self._swipe_start
            if rms < self._offset_thr:
                self._state = _State.OFFSET_WAIT
                self._offset_count = 1
            elif elapsed > _MAX_DURATION_SEC:
                return self._finalize()

        elif self._state == _State.OFFSET_WAIT:
            self._buffer.append(filtered.copy())
            if rms < self._offset_thr:
                self._offset_count += 1
                if self._offset_count >= _OFFSET_CONFIRM_CHUNKS:
                    return self._finalize()
            else:
                self._state = _State.COLLECTING
                self._offset_count = 0

        return None

    def _finalize(self) -> Optional[SwipeEvent]:
        audio = np.concatenate(self._buffer)
        duration = len(audio) / self._sr

        self._state = _State.IDLE
        self._buffer = []
        self._pre_roll.clear()
        self._offset_count = 0

        if duration < _MIN_DURATION_SEC:
            return None

        peak_rms = self._compute_peak_rms(audio)
        self._swipe_index += 1
        wav_path = str(self._save_dir / f"swipe_{self._swipe_index:03d}.wav")
        save_wav(wav_path, audio, self._sr)

        return SwipeEvent(
            audio=audio,
            duration_sec=duration,
            peak_rms=peak_rms,
            timestamp=time.monotonic(),
            wav_path=wav_path,
        )

    def reset(self) -> None:
        self._state = _State.IDLE
        self._buffer = []
        self._pre_roll.clear()
        self._offset_count = 0
        self._zi = np.zeros((self._sos.shape[0], 2))

    @staticmethod
    def _compute_peak_rms(audio: np.ndarray, window: int = 256) -> float:
        n = len(audio)
        if n < window:
            return float(np.sqrt(np.mean(audio ** 2)))
        peaks = [
            float(np.sqrt(np.mean(audio[i:i + window] ** 2)))
            for i in range(0, n - window, window // 2)
        ]
        return max(peaks)
