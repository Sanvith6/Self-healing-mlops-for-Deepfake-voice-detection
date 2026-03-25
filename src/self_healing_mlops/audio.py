from __future__ import annotations

import cmath
import struct
import wave
from pathlib import Path
from typing import Sequence


def read_wav_mono(path: str | Path) -> tuple[list[float], int]:
    """Read a WAV file and return normalized mono samples and sample rate."""
    wav_path = Path(path)
    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        raw_frames = wf.readframes(frame_count)

    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV files are supported")

    total_samples = frame_count * channels
    unpacked = struct.unpack("<" + "h" * total_samples, raw_frames)

    if channels == 1:
        mono = unpacked
    else:
        mono = []
        for idx in range(0, total_samples, channels):
            frame = unpacked[idx : idx + channels]
            mono.append(sum(frame) / channels)

    normalized = [float(s) / 32768.0 for s in mono]
    return normalized, sample_rate


def _zero_crossing_rate(samples: Sequence[float]) -> float:
    if len(samples) < 2:
        return 0.0
    crossings = 0
    for left, right in zip(samples, samples[1:]):
        if (left >= 0 > right) or (left < 0 <= right):
            crossings += 1
    return crossings / (len(samples) - 1)


def _spectral_centroid(samples: Sequence[float], sample_rate: int, n_fft: int = 256) -> float:
    if not samples:
        return 0.0

    window = list(samples[:n_fft])
    if len(window) < n_fft:
        window.extend([0.0] * (n_fft - len(window)))

    magnitudes: list[float] = []
    freqs: list[float] = []
    half = n_fft // 2
    for k in range(half + 1):
        acc = 0j
        for n, sample in enumerate(window):
            acc += sample * cmath.exp(-2j * cmath.pi * k * n / n_fft)
        magnitude = abs(acc)
        magnitudes.append(magnitude)
        freqs.append((k * sample_rate) / n_fft)

    mag_sum = sum(magnitudes)
    if mag_sum == 0:
        return 0.0
    return sum(freq * mag for freq, mag in zip(freqs, magnitudes)) / mag_sum


def extract_features(samples: Sequence[float], sample_rate: int) -> dict[str, float]:
    """Extract a compact baseline feature set from audio samples."""
    if not samples:
        return {
            "duration": 0.0,
            "mean_abs_amplitude": 0.0,
            "rms": 0.0,
            "peak_amplitude": 0.0,
            "zero_crossing_rate": 0.0,
            "spectral_centroid": 0.0,
        }

    abs_values = [abs(v) for v in samples]
    duration = len(samples) / sample_rate if sample_rate else 0.0
    mean_abs = sum(abs_values) / len(abs_values)
    rms = (sum(v * v for v in samples) / len(samples)) ** 0.5
    peak = max(abs_values)

    return {
        "duration": duration,
        "mean_abs_amplitude": mean_abs,
        "rms": rms,
        "peak_amplitude": peak,
        "zero_crossing_rate": _zero_crossing_rate(samples),
        "spectral_centroid": _spectral_centroid(samples, sample_rate),
    }


def extract_features_from_wav(path: str | Path) -> dict[str, float]:
    samples, sample_rate = read_wav_mono(path)
    return extract_features(samples, sample_rate)
