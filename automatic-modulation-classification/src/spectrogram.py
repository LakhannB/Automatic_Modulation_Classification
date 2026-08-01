"""STFT spectrogram generation and augmentation for IQ frames."""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal


def stft_spectrogram(
    iq: np.ndarray,
    n_fft: int = 128,
    hop_length: int = 32,
    window: str = "hann",
) -> np.ndarray:
    """Log-magnitude STFT of a complex IQ frame, DC-centered."""
    _, _, zxx = sp_signal.stft(
        iq,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        window=window,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    mag = np.fft.fftshift(np.abs(zxx), axes=0)
    return (20.0 * np.log10(mag + 1e-12)).astype(np.float32)


def resize(spectrogram: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    h, w = size
    src_h, src_w = spectrogram.shape
    rows = np.linspace(0, src_h - 1, h).round().astype(int)
    cols = np.linspace(0, src_w - 1, w).round().astype(int)
    return spectrogram[np.ix_(rows, cols)].astype(np.float32)


def normalize(spectrogram: np.ndarray) -> np.ndarray:
    return ((spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)).astype(np.float32)


def complex_stft(
    iq: np.ndarray, n_fft: int = 128, hop_length: int = 32, window: str = "hann"
) -> np.ndarray:
    """Complex STFT (DC-centered) retaining phase information."""
    _, _, zxx = sp_signal.stft(
        iq,
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        window=window,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    return np.fft.fftshift(zxx, axes=0)


def iq_to_spectrogram(
    iq: np.ndarray,
    n_fft: int = 128,
    hop_length: int = 32,
    image_size: tuple[int, int] = (128, 128),
    mode: str = "magnitude",
) -> np.ndarray:
    """IQ frame -> normalized CNN input of shape (C, H, W).

    ``magnitude`` gives a single log-magnitude channel; ``complex`` adds the
    real and imaginary STFT channels, which retain the phase information
    needed to separate BPSK / QPSK / 16QAM.
    """
    if mode == "magnitude":
        spec = normalize(resize(stft_spectrogram(iq, n_fft, hop_length), image_size))
        return spec[None, ...]

    if mode == "complex":
        zxx = complex_stft(iq, n_fft, hop_length)
        magnitude = 20.0 * np.log10(np.abs(zxx) + 1e-12)
        channels = [
            normalize(resize(magnitude.astype(np.float32), image_size)),
            normalize(resize(zxx.real.astype(np.float32), image_size)),
            normalize(resize(zxx.imag.astype(np.float32), image_size)),
        ]
        return np.stack(channels).astype(np.float32)

    if mode == "iq":
        return np.stack([iq.real, iq.imag]).astype(np.float32)

    raise ValueError(f"Unknown spectrogram mode '{mode}'")


def input_channels(mode: str) -> int:
    return {"magnitude": 1, "complex": 3, "iq": 2}[mode]


def augment_iq(iq_channels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Time shift, random phase rotation and jitter for raw (2, N) I/Q input."""
    out = np.roll(iq_channels, int(rng.integers(-64, 65)), axis=-1)

    phi = rng.uniform(0, 2 * np.pi)
    i, q = out[0], out[1]
    out = np.stack([i * np.cos(phi) - q * np.sin(phi), i * np.sin(phi) + q * np.cos(phi)])

    if rng.random() < 0.3:
        out = out + rng.standard_normal(out.shape).astype(np.float32) * 0.02
    return out.astype(np.float32)


def augment(spectrogram: np.ndarray, rng: np.random.Generator, mode: str = "magnitude") -> np.ndarray:
    """Time/frequency shifts, SpecAugment masking and additive jitter.

    Works on (H, W) and (C, H, W) spectrograms; raw (2, N) I/Q uses ``augment_iq``.
    """
    if mode == "iq":
        return augment_iq(spectrogram, rng)

    out = spectrogram

    if rng.random() < 0.5:
        out = np.roll(out, int(rng.integers(-16, 17)), axis=-1)
    if rng.random() < 0.5:
        out = np.roll(out, int(rng.integers(-6, 7)), axis=-2)

    if rng.random() < 0.5:
        out = out.copy()
        h, w = out.shape[-2], out.shape[-1]
        t = int(rng.integers(1, 13))
        t0 = int(rng.integers(0, max(1, w - t)))
        out[..., :, t0:t0 + t] = 0.0
        f = int(rng.integers(1, 13))
        f0 = int(rng.integers(0, max(1, h - f)))
        out[..., f0:f0 + f, :] = 0.0

    if rng.random() < 0.3:
        out = out + rng.standard_normal(out.shape).astype(np.float32) * 0.05

    return out.astype(np.float32)
