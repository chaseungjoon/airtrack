"""Microphone capture with queue-based chunk delivery."""

from __future__ import annotations

import queue
import time
from typing import Optional

import numpy as np
import sounddevice as sd

SAMPLE_RATE: int = 44100
CHUNK_SIZE: int = 512   # ~11.6 ms per chunk — fast enough for onset detection
CHANNELS: int = 1
DTYPE: str = "float32"


class MicRecorder:
    """Captures microphone audio and delivers float32 chunks via an internal queue.

    Args:
        sample_rate: Recording sample rate in Hz.
        chunk_size: Samples per callback delivery.
        device: PortAudio device index, or None for the system default.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_size: int = CHUNK_SIZE,
        device: Optional[int] = None,
    ) -> None:
        self._sr = sample_rate
        self._chunk = chunk_size
        self._device = device
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self._stream: Optional[sd.InputStream] = None

    @property
    def sample_rate(self) -> int:
        return self._sr

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        self._queue.put(indata[:, 0].copy())

    def start(self) -> None:
        """Open and start the audio input stream."""
        self._stream = sd.InputStream(
            samplerate=self._sr,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=self._chunk,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Stop and close the audio input stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self) -> Optional[np.ndarray]:
        """Non-blocking read of the next audio chunk."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def measure_noise_floor(self, duration_sec: float = 1.5) -> float:
        """Record silence and return the ambient RMS noise level."""
        chunks: list[np.ndarray] = []
        deadline = time.monotonic() + duration_sec
        while time.monotonic() < deadline:
            chunk = self.read()
            if chunk is not None:
                chunks.append(chunk)
            else:
                time.sleep(0.003)
        if not chunks:
            return 0.005
        return float(np.sqrt(np.mean(np.concatenate(chunks) ** 2)))
