"""Structured robustness benchmarking under noise, interference and frequency offset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .modulation import MODULATIONS, ChannelParams, generate_frame
from .models import build_model
from .spectrogram import input_channels, iq_to_spectrogram


def build_condition_set(
    condition: str,
    values: list[float],
    n_per_class: int,
    frame_length: int,
    seed: int,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Generate one evaluation set per stress value for the requested condition."""
    rng = np.random.default_rng(seed)
    sets: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for value in values:
        iq = np.empty((n_per_class * len(MODULATIONS), frame_length), dtype=np.complex64)
        labels = np.empty(iq.shape[0], dtype=np.int64)

        for class_idx, modulation in enumerate(MODULATIONS):
            for i in range(n_per_class):
                if condition == "snr":
                    params = ChannelParams(snr_db=value)
                elif condition == "interference":
                    params = ChannelParams(snr_db=15.0, interference_db=value)
                elif condition == "freq_offset":
                    params = ChannelParams(snr_db=15.0, freq_offset=value)
                elif condition == "phase_noise":
                    params = ChannelParams(snr_db=15.0, phase_noise_std=value)
                else:
                    raise ValueError(f"Unknown condition '{condition}'")

                row = class_idx * n_per_class + i
                iq[row] = generate_frame(modulation, frame_length, rng, params)
                labels[row] = class_idx

        sets[value] = (iq, labels)

    return sets


@torch.no_grad()
def score(
    model: torch.nn.Module,
    iq: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
    image_size: int = 128,
    mode: str = "complex",
) -> tuple[float, float]:
    model.eval()
    preds = []
    for start in range(0, iq.shape[0], batch_size):
        chunk = iq[start:start + batch_size]
        spectrograms = np.stack(
            [iq_to_spectrogram(frame, image_size=(image_size, image_size), mode=mode)
             for frame in chunk]
        )
        x = torch.from_numpy(spectrograms).to(device)
        preds.append(model(x).argmax(dim=1).cpu().numpy())

    y_pred = np.concatenate(preds)
    return (
        float(accuracy_score(labels, y_pred)),
        float(f1_score(labels, y_pred, average="macro", zero_division=0)),
    )


CONDITIONS = {
    "snr": [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0],
    "interference": [-20.0, -15.0, -10.0, -5.0, 0.0],
    "freq_offset": [0.0, 0.005, 0.01, 0.02, 0.05],
    "phase_noise": [0.0, 0.005, 0.01, 0.02, 0.05],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness benchmark for a trained model")
    parser.add_argument("--checkpoint", default="artifacts/iq_cnn/best_model.pt")
    parser.add_argument("--arch", default="iq_cnn")
    parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    parser.add_argument("--n-per-class", type=int, default=60)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--input-mode", default="iq", choices=["magnitude", "complex", "iq"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="artifacts/robustness.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        args.arch, len(MODULATIONS), input_channels(args.input_mode)
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    results: dict[str, dict[str, dict[str, float]]] = {}

    for condition in args.conditions:
        values = CONDITIONS[condition]
        sets = build_condition_set(condition, values, args.n_per_class,
                                   args.frame_length, args.seed)
        results[condition] = {}
        for value, (iq, labels) in sets.items():
            accuracy, macro_f1 = score(model, iq, labels, device,
                                       image_size=args.image_size, mode=args.input_mode)
            results[condition][str(value)] = {"accuracy": accuracy, "macro_f1": macro_f1}
            print(f"{condition}={value}: accuracy={accuracy:.4f} macro_f1={macro_f1:.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
