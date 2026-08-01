"""Single-frame inference on an IQ file or a synthetic demo frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .modulation import MODULATIONS, ChannelParams, generate_frame
from .models import build_model
from .spectrogram import input_channels, iq_to_spectrogram


def load_iq(path: str | Path, frame_length: int) -> np.ndarray:
    """Load IQ samples from .npy (complex) or raw interleaved float32 .bin."""
    path = Path(path)
    if path.suffix == ".npy":
        iq = np.load(path).astype(np.complex64).ravel()
    else:
        raw = np.fromfile(path, dtype=np.float32)
        iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    if iq.size < frame_length:
        raise ValueError(f"Need at least {frame_length} samples, got {iq.size}")
    return iq[:frame_length]


def predict(model: torch.nn.Module, iq: np.ndarray, classes: list[str],
            device: torch.device, image_size: int = 128, mode: str = "complex") -> dict:
    spectrogram = iq_to_spectrogram(iq, image_size=(image_size, image_size), mode=mode)
    x = torch.from_numpy(spectrogram).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].cpu().numpy()

    top = int(probs.argmax())
    return {
        "prediction": classes[top],
        "confidence": float(probs[top]),
        "probabilities": {c: float(p) for c, p in zip(classes, probs)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a single IQ frame")
    parser.add_argument("--checkpoint", default="artifacts/iq_cnn/best_model.pt")
    parser.add_argument("--arch", default="iq_cnn")
    parser.add_argument("--iq-file", help=".npy (complex) or raw interleaved float32 .bin")
    parser.add_argument("--demo", choices=MODULATIONS,
                        help="classify a freshly generated synthetic frame instead")
    parser.add_argument("--snr-db", type=float, default=12.0)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--input-mode", default="iq", choices=["magnitude", "complex", "iq"])
    args = parser.parse_args()

    if not args.iq_file and not args.demo:
        parser.error("provide --iq-file or --demo <MODULATION>")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        args.arch, len(MODULATIONS), input_channels(args.input_mode)
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.iq_file:
        iq = load_iq(args.iq_file, args.frame_length)
    else:
        rng = np.random.default_rng()
        iq = generate_frame(args.demo, args.frame_length, rng,
                            ChannelParams(snr_db=args.snr_db))

    result = predict(model, iq, MODULATIONS, device, args.image_size, args.input_mode)
    if args.demo:
        result["ground_truth"] = args.demo
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
