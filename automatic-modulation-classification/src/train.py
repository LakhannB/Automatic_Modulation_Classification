"""Training entry point with LR scheduling and per-architecture metrics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, StepLR

from .dataset import build_dataloaders
from .evaluate import evaluate, predict
from .models import build_model, count_parameters
from .spectrogram import input_channels


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_scheduler(name: str, optimizer, epochs: int, steps_per_epoch: int):
    name = name.lower()
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs), "epoch"
    if name == "step":
        return StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.3), "epoch"
    if name == "onecycle":
        max_lr = optimizer.param_groups[0]["lr"]
        return OneCycleLR(optimizer, max_lr=max_lr, epochs=epochs,
                          steps_per_epoch=steps_per_epoch), "step"
    if name == "none":
        return None, "epoch"
    raise ValueError(f"Unknown scheduler '{name}'")


def train_one_epoch(model, loader, criterion, optimizer, device, scheduler, sched_interval) -> float:
    model.train()
    running = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None and sched_interval == "step":
            scheduler.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a modulation classifier")
    parser.add_argument("--arch", default="iq_cnn",
                        choices=["baseline_cnn", "resnet18", "efficientnet_b0", "iq_cnn"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", default="cosine",
                        choices=["cosine", "step", "onecycle", "none"])
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--n-per-class", type=int, default=500)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--n-fft", type=int, default=128)
    parser.add_argument("--hop-length", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--input-mode", default="iq", choices=["magnitude", "complex", "iq"],
                        help="'complex' adds real/imag STFT channels (keeps phase information)")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--data-cache", default="data/amc_dataset.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / args.arch
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, modulations, test_snrs = build_dataloaders(
        n_per_class=args.n_per_class,
        frame_length=args.frame_length,
        batch_size=args.batch_size,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        image_size=(args.image_size, args.image_size),
        mode=args.input_mode,
        num_workers=args.num_workers,
        seed=args.seed,
        cache_path=args.data_cache,
    )

    model = build_model(
        args.arch, len(modulations), input_channels(args.input_mode), args.pretrained
    ).to(device)
    print(f"device={device} arch={args.arch} input_mode={args.input_mode} "
          f"params={count_parameters(model):,}")
    print(f"classes={modulations} train={len(train_loader.dataset)} "
          f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler, sched_interval = build_scheduler(
        args.scheduler, optimizer, args.epochs, len(train_loader)
    )

    history, best_f1 = [], -1.0
    checkpoint_path = output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device,
                               scheduler, sched_interval)
        if scheduler is not None and sched_interval == "epoch":
            scheduler.step()

        y_pred, y_true = predict(model, val_loader, device)
        val_acc = float(accuracy_score(y_true, y_pred))
        val_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        history.append({"epoch": epoch, "train_loss": loss, "val_accuracy": val_acc,
                        "val_macro_f1": val_f1,
                        "lr": optimizer.param_groups[0]["lr"]})
        print(f"epoch {epoch:03d} | loss {loss:.4f} | val_acc {val_acc:.4f} "
              f"| val_f1 {val_f1:.4f} | {time.time() - start:.1f}s")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_path)

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    (output_dir / "classes.json").write_text(json.dumps(modulations, indent=2))

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    metrics = evaluate(model, test_loader, modulations, device, output_dir,
                       split="test", snrs=test_snrs)
    metrics["arch"] = args.arch
    metrics["input_mode"] = args.input_mode
    metrics["params"] = count_parameters(model)
    (output_dir / "summary.json").write_text(json.dumps(
        {k: v for k, v in metrics.items() if k != "report"}, indent=2))
    print(f"\nBest checkpoint: {checkpoint_path} (val macro-F1 {best_f1:.4f})")


if __name__ == "__main__":
    main()
