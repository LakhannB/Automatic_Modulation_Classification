"""Datasets and dataloaders for automatic modulation classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .modulation import generate_dataset
from .spectrogram import augment, iq_to_spectrogram


class ModulationSpectrogramDataset(Dataset):
    """IQ frames converted to STFT spectrograms on the fly."""

    def __init__(
        self,
        iq: np.ndarray,
        labels: np.ndarray,
        snrs: np.ndarray | None = None,
        n_fft: int = 128,
        hop_length: int = 32,
        image_size: tuple[int, int] = (128, 128),
        mode: str = "iq",
        train: bool = False,
        seed: int = 0,
    ) -> None:
        self.iq = iq
        self.labels = labels
        self.snrs = snrs
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.image_size = image_size
        self.mode = mode
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.iq.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        spectrogram = iq_to_spectrogram(
            self.iq[idx], self.n_fft, self.hop_length, self.image_size, self.mode
        )
        if self.train:
            spectrogram = augment(spectrogram, self.rng, self.mode)
        return (
            torch.from_numpy(np.ascontiguousarray(spectrogram)),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def stratified_split(
    labels: np.ndarray, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for class_id in np.unique(labels):
        idx = np.flatnonzero(labels == class_id)
        rng.shuffle(idx)
        n_test = int(round(idx.size * test_ratio))
        n_val = int(round(idx.size * val_ratio))
        test_idx.append(idx[:n_test])
        val_idx.append(idx[n_test:n_test + n_val])
        train_idx.append(idx[n_test + n_val:])

    return (
        rng.permutation(np.concatenate(train_idx)),
        rng.permutation(np.concatenate(val_idx)),
        rng.permutation(np.concatenate(test_idx)),
    )


def load_or_generate(
    n_per_class: int,
    frame_length: int,
    seed: int,
    cache_path: str | Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    cache_path = Path(cache_path) if cache_path else None
    if cache_path is not None and cache_path.exists():
        blob = np.load(cache_path, allow_pickle=True)
        return blob["iq"], blob["labels"], blob["snrs"], list(blob["modulations"])

    iq, labels, snrs, modulations = generate_dataset(
        n_per_class=n_per_class, frame_length=frame_length, seed=seed
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path, iq=iq, labels=labels, snrs=snrs, modulations=np.array(modulations)
        )
    return iq, labels, snrs, modulations


def build_dataloaders(
    n_per_class: int = 500,
    frame_length: int = 1024,
    batch_size: int = 64,
    n_fft: int = 128,
    hop_length: int = 32,
    image_size: tuple[int, int] = (128, 128),
    mode: str = "iq",
    num_workers: int = 0,
    seed: int = 42,
    cache_path: str | Path | None = "data/amc_dataset.npz",
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], np.ndarray]:
    """Return train/val/test dataloaders plus class names and test-set SNRs."""
    iq, labels, snrs, modulations = load_or_generate(
        n_per_class, frame_length, seed, cache_path
    )
    train_idx, val_idx, test_idx = stratified_split(labels, seed=seed)

    def make(indices: np.ndarray, train: bool) -> DataLoader:
        dataset = ModulationSpectrogramDataset(
            iq[indices], labels[indices], snrs[indices],
            n_fft, hop_length, image_size, mode, train=train, seed=seed,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers)

    return (
        make(train_idx, True),
        make(val_idx, False),
        make(test_idx, False),
        modulations,
        snrs[test_idx],
    )
