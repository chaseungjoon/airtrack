#!/usr/bin/env python3
"""
AirTrack — Active Acoustic Sonar PoC  (§3.3)
==============================================
Validates the ultrasonic sonar concept from §3.3 of PLAN.md:
  the built-in speaker emits a ~20 kHz carrier; the microphone records
  the reflection; Doppler shifts + acoustic cross-section (finger count)
  are extracted via I/Q demodulation.

Three phases (all run by default):

  check      Phase 1 — hardware validation: play carrier, FFT recording,
                       confirm carrier peak + SNR
  visualize  Phase 2 — live spectrum plot around carrier band: visually
                       confirm Doppler shifts when hand moves
  collect    Phase 3 — data collection: baseline vs 1/2/3-finger swipes,
                       compare demodulated RMS to validate cross-section effect

Usage:
    python scripts/mvp_sonar_test.py                    # all three phases
    python scripts/mvp_sonar_test.py --mode check
    python scripts/mvp_sonar_test.py --mode visualize
    python scripts/mvp_sonar_test.py --mode collect
    python scripts/mvp_sonar_test.py --carrier 18000    # lower if 20kHz fails
    python scripts/mvp_sonar_test.py --list-devices
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import sounddevice as sd
    from scipy.signal import stft as scipy_stft  # noqa: F401  (available for ad-hoc use)
except ImportError as exc:
    print(f"[ERROR] Missing dependency: {exc}")
    print("Run:  pip install sounddevice scipy numpy matplotlib")
    sys.exit(1)

from airtrack.contact.sonar import SonarTransceiver


# ── display helpers ─────────────────────────────────────────────────────────

def _header(msg: str) -> None:
    print(f"\n{'─' * 62}")
    print(f"  {msg}")
    print(f"{'─' * 62}")


def _bar(value: float, max_val: float, width: int = 20) -> str:
    filled = int(round(min(value / max(max_val, 1e-12), 1.0) * width))
    return "█" * filled + "░" * (width - filled)


# ── Phase 1: Hardware Check ──────────────────────────────────────────────────

def phase_check(carrier_hz: float, device, sr: int) -> bool:
    """Play carrier for 1.5 s, FFT the recording, verify peak + SNR."""
    _header("Phase 1 — Hardware Check")
    print(f"  Playing {carrier_hz:.0f} Hz carrier for 1.5 seconds — stay quiet.\n")

    n = int(1.5 * sr)
    t = np.arange(n) / sr
    carrier = (0.7 * np.sin(2.0 * np.pi * carrier_hz * t)).astype(np.float32)

    try:
        recording = sd.playrec(carrier, samplerate=sr, channels=1, device=device, dtype="float32")
        sd.wait()
    except Exception as exc:
        print(f"  [ERROR] Audio I/O failed: {exc}")
        print("  Use --list-devices to select a device, or --device N to specify one.")
        return False

    rec = recording[:, 0]
    freqs = np.fft.rfftfreq(len(rec), 1.0 / sr)
    mag = np.abs(np.fft.rfft(rec))

    band = (freqs >= carrier_hz - 500) & (freqs <= carrier_hz + 500)
    outside = ~band & (freqs > 100)  # exclude DC for noise floor estimate

    if not np.any(band):
        print(f"  [!] {carrier_hz:.0f} Hz exceeds Nyquist ({sr // 2} Hz) — lower --carrier or raise --sr.")
        return False

    peak_freq = freqs[band][np.argmax(mag[band])]
    peak_mag = float(np.max(mag[band]))
    noise_floor = float(np.median(mag[outside])) if np.any(outside) else 1e-8
    snr_db = 20.0 * np.log10(peak_mag / (noise_floor + 1e-8))

    print(f"  Peak detected at : {peak_freq:.1f} Hz  (target {carrier_hz:.0f} Hz, error ±{abs(peak_freq - carrier_hz):.1f} Hz)")
    print(f"  SNR              : {snr_db:.1f} dB")

    if snr_db >= 10:
        print(f"\n  ✓  Carrier confirmed at {snr_db:.0f} dB SNR — hardware is suitable.\n")
        return True
    elif snr_db >= 3:
        print(f"\n  ~  Marginal SNR ({snr_db:.1f} dB). Continuing, but results may be noisy.")
        print(f"     Try --carrier 18000 or 19000 for better speaker/mic response.\n")
        return True
    else:
        print(f"\n  ✗  Carrier not detected (SNR = {snr_db:.1f} dB). Troubleshooting:")
        print(f"     • Raise system volume to ≥ 70%")
        print(f"     • Try --carrier 18000 (better MacBook response at 18 kHz)")
        print(f"     • Disable System Settings > Sound > Input noise reduction")
        print(f"     • Try --device to select the correct I/O device\n")
        return False


# ── Phase 2: Live Visualization ──────────────────────────────────────────────

def phase_visualize(transceiver: SonarTransceiver, carrier_hz: float, sr: int, duration_sec: float) -> None:
    """Live spectrum + RMS plot. Shows Doppler shifts and cross-section energy."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [!] matplotlib not installed — skipping visualization.")
        print("  Run: pip install matplotlib\n")
        return

    _header("Phase 2 — Live Doppler Visualization")
    print(f"  Watch the spectrum around {carrier_hz / 1000:.0f} kHz.")
    print(f"  Wave your hand slowly over the keyboard — look for sidebands and RMS spikes.")
    print(f"  Close the window or press Ctrl+C to continue.\n")

    # Raw rolling buffer: 0.5 s of audio for FFT
    buf_len = sr // 2
    raw_buf = np.zeros(buf_len, dtype=np.float32)
    rms_history: list[float] = []
    start_t = time.monotonic()

    fig, (ax_fft, ax_rms) = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle(f"AirTrack Sonar PoC  —  {carrier_hz / 1000:.0f} kHz carrier")
    plt.ion()
    plt.tight_layout(pad=2.0)
    plt.show()

    # Frames per block at 44100 Hz with block_size=4096 → ~10.7 frames/sec
    # Request only the last ~2 seconds worth on each iteration (no explicit clear —
    # the deque maxlen discards stale frames automatically, avoiding starvation).
    _frames_per_sec = transceiver.sample_rate / transceiver.block_size
    _vis_window = max(5, int(2.0 * _frames_per_sec))

    try:
        while plt.fignum_exists(fig.number) and time.monotonic() - start_t < duration_sec:
            frames = transceiver.get_frames(n=_vis_window)

            if frames:
                rms_history.extend(f.doppler_rms for f in frames)
                rms_history = rms_history[-300:]

                new_raw = np.concatenate([f.raw for f in frames])
                n = min(len(new_raw), buf_len)
                raw_buf = np.roll(raw_buf, -n)
                raw_buf[-n:] = new_raw[-n:]

            # Spectrum around carrier
            freqs = np.fft.rfftfreq(buf_len, 1.0 / sr)
            mag_db = 20.0 * np.log10(np.abs(np.fft.rfft(raw_buf)) + 1e-8)
            band = (freqs >= carrier_hz - 400) & (freqs <= carrier_hz + 400)

            ax_fft.clear()
            if np.any(band):
                ax_fft.plot(freqs[band] - carrier_hz, mag_db[band], color="lime", linewidth=1)
                ax_fft.axvline(0, color="r", linestyle="--", alpha=0.6, label=f"Carrier  {carrier_hz:.0f} Hz")
                ax_fft.set_xlabel("Offset from carrier (Hz)")
                ax_fft.set_ylabel("Power (dB)")
                ax_fft.set_title("Carrier band — Doppler sidebands appear on hand motion")
                ax_fft.legend(fontsize=8, loc="upper right")

            # Demodulated RMS over time
            ax_rms.clear()
            if rms_history:
                t_ax = np.arange(len(rms_history)) * transceiver.block_size / sr
                ax_rms.plot(t_ax, rms_history, color="cyan", linewidth=1)
                if len(rms_history) > 10:
                    baseline = np.percentile(rms_history, 10)
                    ax_rms.axhline(baseline, color="yellow", linestyle=":", alpha=0.7, label="10th pct (quiet)")
                    ax_rms.legend(fontsize=8)
                ax_rms.set_xlabel("Time (s)")
                ax_rms.set_ylabel("Doppler RMS")
                ax_rms.set_title("Acoustic cross-section energy — higher = more / larger contact area")

            plt.tight_layout(pad=2.0)
            plt.pause(0.15)

    except KeyboardInterrupt:
        pass

    if plt.fignum_exists(fig.number):
        plt.close(fig)
    print("  Visualization complete.\n")


# ── Phase 3: Data Collection ─────────────────────────────────────────────────

def phase_collect(transceiver: SonarTransceiver, n_trials: int, trial_sec: float) -> None:
    """Record per-class swipes and validate monotonic RMS ordering."""
    _header("Phase 3 — Cross-Section Energy Validation")
    print(f"  {n_trials} trial(s) × 4 classes, {trial_sec:.0f} s per trial.")
    print(f"  Goal: confirm RMS increases monotonically from baseline → 3 fingers.\n")

    classes: list[tuple[int, str, str]] = [
        (0, "baseline",  "Keep hands still — do NOT touch keyboard"),
        (1, "1 finger",  "Swipe ONE finger across the keys, steady pace"),
        (2, "2 fingers", "Swipe TWO fingers across the keys, steady pace"),
        (3, "3 fingers", "Swipe THREE fingers across the keys, steady pace"),
        (4, "4 fingers", "Swipe FOUR fingers across the keys, steady pace"),
    ]

    results: dict[str, list[float]] = {label: [] for _, label, _ in classes}
    peak_scale = 1e-4  # will be updated for the bar display

    for _, label, instruction in classes:
        print(f"  ── {label.upper()} ──")
        print(f"  {instruction}\n")

        for trial in range(1, n_trials + 1):
            print(f"  Trial {trial}/{n_trials}  Press Enter to start...", end="", flush=True)
            try:
                input()
            except KeyboardInterrupt:
                print("\n  Interrupted.")
                return

            transceiver.clear_buffer()

            for remaining in range(int(trial_sec), 0, -1):
                sys.stdout.write(f"\r  Recording... {remaining}s   ")
                sys.stdout.flush()
                time.sleep(1.0)

            frames = transceiver.get_frames()
            if not frames:
                sys.stdout.write("\r  [!] No frames captured.\n")
                continue

            rms_vals = [f.doppler_rms for f in frames]
            mean_rms = float(np.mean(rms_vals))
            peak_rms = float(np.max(rms_vals))
            results[label].append(mean_rms)
            peak_scale = max(peak_scale, peak_rms)

            bar = _bar(mean_rms, peak_scale)
            sys.stdout.write(f"\r  Trial {trial}  mean={mean_rms:.6f}  |{bar}|\n")
            sys.stdout.flush()

        print()

    # ── results table ─────────────────────────────────────────────────────────
    _header("Results")
    print(f"\n  {'Class':<14}  {'Mean RMS':>12}  {'Std':>10}  {'N':>3}")
    print(f"  {'─' * 14}  {'─' * 12}  {'─' * 10}  {'─' * 3}")

    class_means: dict[str, float] = {}
    for _, label, _ in classes:
        vals = results[label]
        if not vals:
            continue
        m = float(np.mean(vals))
        s = float(np.std(vals))
        class_means[label] = m
        print(f"  {label:<14}  {m:>12.6f}  {s:>10.6f}  {len(vals):>3}")

    ordered = ["baseline", "1 finger", "2 fingers", "3 fingers"]
    available = [l for l in ordered if l in class_means]
    monotonic = all(
        class_means[available[i]] < class_means[available[i + 1]]
        for i in range(len(available) - 1)
    )

    print(f"\n  Monotonic ordering (baseline → 3f): {'✓ YES' if monotonic else '✗ NO'}")

    # Per-pair significance (Mann-Whitney U if scipy available)
    if len(available) >= 2:
        try:
            from scipy.stats import mannwhitneyu
            print()
            for i in range(len(available) - 1):
                a, b = available[i], available[i + 1]
                a_vals, b_vals = results[a], results[b]
                if len(a_vals) >= 2 and len(b_vals) >= 2:
                    _, p = mannwhitneyu(a_vals, b_vals, alternative="less")
                    flag = "✓ p<0.05" if p < 0.05 else f"p={p:.3f} (need more trials)"
                    print(f"  {a:<14} < {b:<14}  {flag}")
        except Exception:
            pass

    print()
    if monotonic:
        print("  ✓  Cross-section effect confirmed — §3.3 sonar is VIABLE.")
        print("     Proceed to integrate SonarTransceiver into the AirTrack pipeline.\n")
    else:
        print("  ~  No clear ordering. Tips:")
        print("     • Increase --trials (try 5) to reduce variance")
        print("     • Keep swipe speed and pressure consistent across classes")
        print("     • Run phase_check first — low SNR degrades discrimination")
        print("     • Move closer to the laptop speakers/mic\n")


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AirTrack — Active Acoustic Sonar PoC (§3.3)"
    )
    parser.add_argument(
        "--mode",
        choices=["check", "visualize", "collect", "all"],
        default="all",
        help="Which phase(s) to run (default: all)",
    )
    parser.add_argument("--carrier", type=float, default=20_000.0,
                        help="Carrier frequency Hz (default 20000; try 18000 if hardware fails)")
    parser.add_argument("--device", type=int, default=None,
                        help="Audio device index for both I/O (see --list-devices)")
    parser.add_argument("--sr", type=int, default=44_100,
                        help="Sample rate Hz (default 44100)")
    parser.add_argument("--trials", type=int, default=3,
                        help="Trials per class in collect mode (default 3)")
    parser.add_argument("--trial-sec", type=float, default=3.0,
                        help="Recording duration per trial in seconds (default 3)")
    parser.add_argument("--viz-sec", type=float, default=30.0,
                        help="Visualization window duration in seconds (default 30)")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        print("\nAudio devices:")
        for i, dev in enumerate(sd.query_devices()):
            inp = dev["max_input_channels"]   # type: ignore[index]
            out = dev["max_output_channels"]  # type: ignore[index]
            if inp > 0 or out > 0:
                print(f"  [{i:2d}]  in={inp} out={out}  {dev['name']}")  # type: ignore[index]
        return

    _header("AirTrack — Active Acoustic Sonar PoC  (§3.3)")
    print(f"  Carrier  : {args.carrier:.0f} Hz")
    print(f"  SR       : {args.sr} Hz  (Nyquist {args.sr // 2} Hz)")

    if args.carrier >= args.sr / 2:
        print(f"\n  [ERROR] Carrier {args.carrier:.0f} Hz ≥ Nyquist {args.sr // 2} Hz.")
        print("  Lower --carrier or raise --sr.")
        sys.exit(1)

    run_check     = args.mode in ("check", "all")
    run_visualize = args.mode in ("visualize", "all")
    run_collect   = args.mode in ("collect", "all")

    # Phase 1 — no transceiver needed (uses sd.playrec)
    if run_check:
        ok = phase_check(args.carrier, args.device, args.sr)
        if not ok and args.mode == "all":
            print("  Hardware check failed — aborting remaining phases.")
            return

    # Phases 2 & 3 share a running transceiver
    if run_visualize or run_collect:
        transceiver = SonarTransceiver(
            carrier_hz=args.carrier,
            sample_rate=args.sr,
            device=args.device,
        )
        try:
            transceiver.start()
        except Exception as exc:
            print(f"\n[ERROR] Could not start transceiver: {exc}")
            print("  Use --list-devices to inspect available devices.")
            sys.exit(1)

        # Let filter states settle before collecting data
        print("\n  Transceiver warming up for 0.5 s...")
        time.sleep(0.5)
        transceiver.clear_buffer()

        try:
            if run_visualize:
                phase_visualize(transceiver, args.carrier, args.sr, args.viz_sec)
            if run_collect:
                phase_collect(transceiver, args.trials, args.trial_sec)
        finally:
            transceiver.stop()

    _header("Done")
    print(f"  §3.3 sonar PoC complete.\n")


if __name__ == "__main__":
    main()
