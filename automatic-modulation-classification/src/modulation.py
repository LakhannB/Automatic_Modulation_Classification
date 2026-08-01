"""IQ waveform generation for digital modulation schemes (BPSK, QPSK, 16QAM, FSK)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODULATIONS = ["BPSK", "QPSK", "16QAM", "FSK"]


@dataclass
class ChannelParams:
    """Channel impairments applied to a generated frame."""

    snr_db: float = 15.0
    freq_offset: float = 0.0        # normalized frequency offset (fraction of fs)
    phase_offset: float = 0.0       # radians
    phase_noise_std: float = 0.0    # random-walk phase noise per sample
    interference_db: float | None = None  # CW interferer power relative to signal (dB)


def _normalize_power(iq: np.ndarray) -> np.ndarray:
    power = float(np.mean(np.abs(iq) ** 2))
    return iq if power <= 0 else iq / np.sqrt(power)


def _rrc_filter(beta: float, sps: int, span: int = 8) -> np.ndarray:
    """Root-raised-cosine pulse-shaping filter taps."""
    n = np.arange(-span * sps / 2, span * sps / 2 + 1)
    t = n / sps
    taps = np.zeros_like(t)

    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            taps[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta > 0 and np.isclose(abs(ti), 1 / (4 * beta)):
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            taps[i] = num / den

    return taps / np.sqrt(np.sum(taps**2))


def _symbols(modulation: str, n_symbols: int, rng: np.random.Generator) -> np.ndarray:
    if modulation == "BPSK":
        bits = rng.integers(0, 2, n_symbols)
        return (2 * bits - 1).astype(np.complex128)

    if modulation == "QPSK":
        idx = rng.integers(0, 4, n_symbols)
        return np.exp(1j * (np.pi / 4 + idx * np.pi / 2))

    if modulation == "16QAM":
        levels = np.array([-3, -1, 1, 3])
        real = rng.choice(levels, n_symbols)
        imag = rng.choice(levels, n_symbols)
        return (real + 1j * imag) / np.sqrt(10)

    raise ValueError(f"{modulation} is not a linear modulation")


def _linear_modulation(
    modulation: str, frame_length: int, sps: int, rng: np.random.Generator, beta: float = 0.35
) -> np.ndarray:
    n_symbols = frame_length // sps + 16
    symbols = _symbols(modulation, n_symbols, rng)

    upsampled = np.zeros(n_symbols * sps, dtype=np.complex128)
    upsampled[::sps] = symbols

    taps = _rrc_filter(beta, sps)
    shaped = np.convolve(upsampled, taps, mode="same")
    return shaped[:frame_length]


def _fsk(frame_length: int, sps: int, rng: np.random.Generator, separation: float = 0.1) -> np.ndarray:
    n_symbols = frame_length // sps + 1
    tones = np.array([-1.5, -0.5, 0.5, 1.5]) * separation
    freqs = np.repeat(rng.choice(tones, n_symbols), sps)[:frame_length]
    phase = 2 * np.pi * np.cumsum(freqs)
    return np.exp(1j * phase)


def add_awgn(iq: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    signal_power = float(np.mean(np.abs(iq) ** 2))
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.standard_normal(iq.size) + 1j * rng.standard_normal(iq.size)
    return iq + noise * np.sqrt(noise_power / 2.0)


def add_cw_interference(
    iq: np.ndarray, interference_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Add a continuous-wave interferer at a random offset frequency."""
    signal_power = float(np.mean(np.abs(iq) ** 2))
    amplitude = np.sqrt(signal_power * 10 ** (interference_db / 10.0))
    freq = rng.uniform(-0.25, 0.25)
    phase = rng.uniform(0, 2 * np.pi)
    t = np.arange(iq.size)
    return iq + amplitude * np.exp(1j * (2 * np.pi * freq * t + phase))


def apply_channel(iq: np.ndarray, params: ChannelParams, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(iq.size)
    out = iq * np.exp(1j * (2 * np.pi * params.freq_offset * t + params.phase_offset))

    if params.phase_noise_std > 0:
        out = out * np.exp(1j * np.cumsum(rng.standard_normal(iq.size) * params.phase_noise_std))

    if params.interference_db is not None:
        out = add_cw_interference(out, params.interference_db, rng)

    return add_awgn(out, params.snr_db, rng)


def generate_frame(
    modulation: str,
    frame_length: int,
    rng: np.random.Generator,
    params: ChannelParams,
    sps: int | None = None,
) -> np.ndarray:
    """Generate one impaired IQ frame for the requested modulation scheme."""
    sps = sps or int(rng.integers(4, 9))

    if modulation == "FSK":
        iq = _fsk(frame_length, sps, rng)
    else:
        iq = _linear_modulation(modulation, frame_length, sps, rng)

    iq = _normalize_power(iq)
    iq = apply_channel(iq, params, rng)
    return _normalize_power(iq).astype(np.complex64)


def generate_dataset(
    n_per_class: int,
    frame_length: int = 1024,
    snr_db_range: tuple[float, float] = (-4.0, 18.0),
    freq_offset_range: float = 0.01,
    modulations: list[str] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build an IQ dataset spanning a range of SNRs.

    Returns (iq, labels, snrs, modulations).
    """
    modulations = modulations or MODULATIONS
    rng = np.random.default_rng(seed)

    total = n_per_class * len(modulations)
    iq = np.empty((total, frame_length), dtype=np.complex64)
    labels = np.empty(total, dtype=np.int64)
    snrs = np.empty(total, dtype=np.float32)

    for class_idx, modulation in enumerate(modulations):
        for i in range(n_per_class):
            row = class_idx * n_per_class + i
            snr_db = float(rng.uniform(*snr_db_range))
            params = ChannelParams(
                snr_db=snr_db,
                freq_offset=float(rng.uniform(-freq_offset_range, freq_offset_range)),
                phase_offset=float(rng.uniform(0, 2 * np.pi)),
                phase_noise_std=0.005,
            )
            iq[row] = generate_frame(modulation, frame_length, rng, params)
            labels[row] = class_idx
            snrs[row] = snr_db

    perm = rng.permutation(total)
    return iq[perm], labels[perm], snrs[perm], modulations
