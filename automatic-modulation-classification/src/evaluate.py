"""Evaluation: classification report, confusion matrix and accuracy-vs-SNR curve."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(dim=1).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


def accuracy_vs_snr(
    y_true: np.ndarray, y_pred: np.ndarray, snrs: np.ndarray, bin_width: float = 2.0
) -> dict[str, float]:
    """Accuracy grouped into SNR bins — the standard AMC reporting curve."""
    if snrs is None:
        return {}
    bins = np.round(np.asarray(snrs) / bin_width) * bin_width
    return {
        f"{edge:+.0f}dB": float(accuracy_score(y_true[bins == edge], y_pred[bins == edge]))
        for edge in np.unique(bins)
        if np.sum(bins == edge) > 0
    }


def save_confusion_matrix(cm: list[list[int]], classes: list[str], path: str | Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return

    cm_array = np.asarray(cm, dtype=float)
    normalized = cm_array / np.clip(cm_array.sum(axis=1, keepdims=True), 1, None)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (row-normalized)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if normalized[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_snr_curve(curve: dict[str, float], path: str | Path) -> None:
    if not curve:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return

    snrs = [float(k.replace("dB", "")) for k in curve]
    order = np.argsort(snrs)
    xs = np.array(snrs)[order]
    ys = np.array(list(curve.values()))[order]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification accuracy vs SNR")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    classes: list[str],
    device: torch.device,
    output_dir: str | Path | None = None,
    split: str = "test",
    snrs: np.ndarray | None = None,
) -> dict:
    y_pred, y_true = predict(model, loader, device)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "report": classification_report(y_true, y_pred, target_names=classes,
                                        zero_division=0, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "accuracy_vs_snr": accuracy_vs_snr(y_true, y_pred, snrs) if snrs is not None else {},
    }

    print(f"\n[{split}] accuracy={metrics['accuracy']:.4f} macro-F1={metrics['macro_f1']:.4f}")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    if metrics["accuracy_vs_snr"]:
        print("accuracy vs SNR:", json.dumps(metrics["accuracy_vs_snr"], indent=2))

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2))
        save_confusion_matrix(metrics["confusion_matrix"], classes,
                              output_dir / f"{split}_confusion_matrix.png")
        save_snr_curve(metrics["accuracy_vs_snr"], output_dir / f"{split}_accuracy_vs_snr.png")

    return metrics


def main() -> None:
    import argparse

    from .dataset import build_dataloaders
    from .models import build_model
    from .spectrogram import input_channels

    parser = argparse.ArgumentParser(description="Evaluate a trained modulation classifier")
    parser.add_argument("--checkpoint", default="artifacts/iq_cnn/best_model.pt")
    parser.add_argument("--arch", default="iq_cnn")
    parser.add_argument("--n-per-class", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-mode", default="iq", choices=["magnitude", "complex", "iq"])
    parser.add_argument("--output-dir", default="artifacts/eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, modulations, test_snrs = build_dataloaders(
        n_per_class=args.n_per_class, batch_size=args.batch_size, mode=args.input_mode
    )
    model = build_model(
        args.arch, len(modulations), input_channels(args.input_mode)
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    evaluate(model, test_loader, modulations, device, args.output_dir,
             split="test", snrs=test_snrs)


if __name__ == "__main__":
    main()
