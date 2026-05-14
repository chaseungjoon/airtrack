"""Modality C — Active acoustic sensing (ultrasonic FMCW sonar) for finger count.

Emits an inaudible ~20 kHz carrier via the built-in speaker and records the
microphone simultaneously (full-duplex). Two measurable effects in the echo:

  Doppler shift  : radial velocity v → Δf = 2·v·f_c / c  (~35 Hz at 30 cm/s)
  Cross-section  : reflective area ∝ number of fingers → higher echo energy

DSP pipeline per block:
  1. Emit  : sin(2π·f_c·n/sr) phase-continuously at carrier_hz
  2. Receive: record mic at same sample rate
  3. I/Q mix: multiply received by cos/sin(phase) → shifts f_c → DC
  4. LPF (500 Hz Butterworth): isolate motion content, discard 2·f_c artefact
  5. Baseband amplitude ∝ acoustic cross-section (finger-count proxy)

Overflow fix: blocksize=4096, default latency. The 23 ms blocks of the old
1024-sample setting left insufficient margin for Python GIL + numpy ops to
complete before the next callback, causing xrun input overflows.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.signal import butter, sosfilt

logger = logging.getLogger(__name__)

_LPF_CUTOFF_HZ = 500
_LPF_ORDER = 6


@dataclass
class SonarFrame:
    """One processed sonar block."""
    raw: np.ndarray        # float32 mono mic samples, shape (block_size,)
    baseband: np.ndarray   # complex64 I+jQ after demod + LPF, shape (block_size,)
    doppler_rms: float     # RMS of |baseband| — acoustic cross-section proxy
    timestamp: float       # time.monotonic()


def make_lpf(cutoff_hz: float = _LPF_CUTOFF_HZ, sample_rate: int = 44_100) -> np.ndarray:
    """Return SOS coefficients for a Butterworth low-pass filter."""
    return butter(_LPF_ORDER, cutoff_hz, fs=sample_rate, btype="low", output="sos")  # type: ignore[return-value]


class SonarTransceiver:
    """Full-duplex ultrasonic transceiver using built-in speaker and microphone.

    Emits a phase-continuous carrier at carrier_hz while demodulating the
    received signal via I/Q mixing. The complex baseband amplitude tracks
    acoustic cross-section — a finger-count proxy independent of swipe force.

    Args:
        carrier_hz: Carrier frequency in Hz (default 20 000 — inaudible to humans).
        sample_rate: Audio sample rate (must satisfy Nyquist > carrier_hz).
        block_size: Frames per audio callback. 4096 is the default; smaller
            values increase time resolution but risk xrun input overflows on
            Python's GIL-constrained callback thread.
        amplitude: Carrier output amplitude 0–1 (default 0.7).
        device: sounddevice device index or (input_idx, output_idx) tuple.
        on_frame: Optional callback invoked per processed frame in audio thread.
        buffer_size: Rolling frame buffer depth.
    """

    def __init__(
        self,
        carrier_hz: float = 20_000.0,
        sample_rate: int = 44_100,
        block_size: int = 4096,
        amplitude: float = 0.7,
        device=None,
        on_frame: Callable[[SonarFrame], None] | None = None,
        buffer_size: int = 500,
    ) -> None:
        self.carrier_hz = carrier_hz
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.amplitude = amplitude

        self._device = device
        self._on_frame = on_frame

        self._lpf_sos = make_lpf(_LPF_CUTOFF_HZ, sample_rate)
        n_sec = self._lpf_sos.shape[0]
        self._zi_I = np.zeros((n_sec, 2))
        self._zi_Q = np.zeros((n_sec, 2))

        self._phase = 0.0  # carrier phase accumulator (radians), bounded to [0, 2π]

        self._buffer: deque[SonarFrame] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._stream = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        import sounddevice as sd
        # Default latency avoids the xrun input overflow that latency="low" causes
        # on Python's GIL-constrained callback thread.
        self._stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            device=self._device,
            channels=(1, 1),
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        logger.info(
            "SonarTransceiver started  f_c=%.0f Hz  sr=%d  block=%d",
            self.carrier_hz, self.sample_rate, self.block_size,
        )

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("SonarTransceiver stopped")

    # ── data access ──────────────────────────────────────────────────────────

    def get_frames(self, n: int | None = None) -> list[SonarFrame]:
        """Return up to n most recent frames (thread-safe copy).

        Prefer NOT calling clear_buffer() in visualization loops — let the
        deque's maxlen silently discard old frames instead. Explicit clears
        on every loop iteration can starve the callback of queue space.
        """
        with self._lock:
            frames = list(self._buffer)
        return frames if n is None else frames[-n:]

    def clear_buffer(self) -> None:
        with self._lock:
            self._buffer.clear()

    # ── audio callback ───────────────────────────────────────────────────────

    def _callback(self, indata, outdata, frames, _time_info, status) -> None:
        if status:
            logger.warning("Stream: %s", status)
        try:
            # Phase ramp for this block (phase-continuous across calls)
            phase_ramp = self._phase + 2.0 * np.pi * self.carrier_hz * np.arange(frames) / self.sample_rate
            outdata[:, 0] = (self.amplitude * np.sin(phase_ramp)).astype(np.float32)
            self._phase = float(
                (self._phase + 2.0 * np.pi * self.carrier_hz * frames / self.sample_rate) % (2.0 * np.pi)
            )

            # I/Q demodulation: shift carrier to DC, Doppler sidebands to ±Δf
            rec = indata[:, 0].astype(np.float64)
            I_mix = rec * np.cos(phase_ramp)
            Q_mix = rec * np.sin(phase_ramp)

            # LPF with persistent state — removes 2·f_c artefact, keeps motion content
            I_filt, self._zi_I = sosfilt(self._lpf_sos, I_mix, zi=self._zi_I)
            Q_filt, self._zi_Q = sosfilt(self._lpf_sos, Q_mix, zi=self._zi_Q)

            baseband = (I_filt + 1j * Q_filt).astype(np.complex64)
            doppler_rms = float(np.sqrt(np.mean(np.abs(baseband) ** 2)))

            frame = SonarFrame(
                raw=rec.astype(np.float32),
                baseband=baseband,
                doppler_rms=doppler_rms,
                timestamp=time.monotonic(),
            )
            with self._lock:
                self._buffer.append(frame)
            if self._on_frame is not None:
                self._on_frame(frame)
        except Exception as exc:
            logger.error("SonarTransceiver callback: %s", exc)
