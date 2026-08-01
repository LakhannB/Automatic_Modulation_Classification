"""CNN architectures compared for modulation classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

ARCHITECTURES = ["baseline_cnn", "resnet18", "efficientnet_b0", "iq_cnn"]


class BaselineCNN(nn.Module):
    """Lightweight reference CNN for single-channel spectrograms."""

    def __init__(self, num_classes: int, in_channels: int = 1, dropout: float = 0.3) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(in_channels, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class IQCNN1D(nn.Module):
    """1D CNN over raw I/Q samples (2 x N), which preserves constellation phase."""

    def __init__(self, num_classes: int, in_channels: int = 2, dropout: float = 0.3) -> None:
        super().__init__()

        def block(cin: int, cout: int, kernel: int = 7) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel, padding=kernel // 2, bias=False),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
                nn.Conv1d(cout, cout, kernel, padding=kernel // 2, bias=False),
                nn.BatchNorm1d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, 64), block(64, 128), block(128, 128), block(128, 256)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def _resnet18(num_classes: int, in_channels: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _efficientnet_b0(num_classes: int, in_channels: int, pretrained: bool) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    old = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
        padding=old.padding, bias=False,
    )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model


def build_model(
    name: str, num_classes: int, in_channels: int = 1, pretrained: bool = False
) -> nn.Module:
    name = name.lower()
    if name in {"baseline_cnn", "cnn"}:
        return BaselineCNN(num_classes, in_channels)
    if name == "resnet18":
        return _resnet18(num_classes, in_channels, pretrained)
    if name in {"efficientnet_b0", "efficientnet"}:
        return _efficientnet_b0(num_classes, in_channels, pretrained)
    if name in {"iq_cnn", "iq"}:
        return IQCNN1D(num_classes, in_channels)
    raise ValueError(f"Unknown architecture '{name}'. Options: {ARCHITECTURES}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
