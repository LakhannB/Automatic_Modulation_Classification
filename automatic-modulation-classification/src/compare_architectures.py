"""Train several architectures under identical settings and compare them."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(arch: str, args: argparse.Namespace) -> dict:
    command = [
        sys.executable, "-m", "src.train",
        "--arch", arch,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--scheduler", args.scheduler,
        "--n-per-class", str(args.n_per_class),
        "--output-dir", args.output_dir,
    ]
    print(f"\n=== Training {arch} ===\n{' '.join(command)}")
    subprocess.run(command, check=True)

    summary_path = Path(args.output_dir) / arch / "summary.json"
    return json.loads(summary_path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare CNN architectures for AMC")
    parser.add_argument("--archs", nargs="*",
                        default=["baseline_cnn", "resnet18", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scheduler", default="cosine")
    parser.add_argument("--n-per-class", type=int, default=500)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()

    summaries = {arch: run(arch, args) for arch in args.archs}

    print("\n=== Comparison ===")
    header = f"{'architecture':<18}{'params':>12}{'accuracy':>12}{'macro-F1':>12}"
    print(header)
    print("-" * len(header))
    for arch, summary in summaries.items():
        print(f"{arch:<18}{summary['params']:>12,}{summary['accuracy']:>12.4f}"
              f"{summary['macro_f1']:>12.4f}")

    output = Path(args.output_dir) / "comparison.json"
    output.write_text(json.dumps(summaries, indent=2))
    print(f"\nSaved -> {output}")


if __name__ == "__main__":
    main()
