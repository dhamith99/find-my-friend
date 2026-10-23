"""Build any of the three models by name, with the same interface."""
from __future__ import annotations

import io

import torch
import torch.nn as nn
from torchvision import models

from src.models.custom_cnn import CustomCNN

MODEL_NAMES = ("custom_cnn", "mobilenet_v3_small", "resnet18")


def build_model(name: str, num_classes: int = 4, dropout: float = 0.2,
                pretrained: bool = True, freeze: str = "none") -> nn.Module:
    """
    freeze:
      "none" - train every layer (fine-tune the whole network)
      "head" - keep the pretrained backbone fixed, train only the new classifier
    """
    if name == "custom_cnn":
        return CustomCNN(num_classes, dropout)          # always from scratch

    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        # classifier = [Linear(576,1024), Hardswish, Dropout, Linear(1024,1000)]
        model.classifier[2] = nn.Dropout(dropout)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        head = model.classifier

    elif name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
        head = model.fc

    else:
        raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")

    if freeze == "head":
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    elif freeze != "none":
        raise ValueError(f"freeze must be 'none' or 'head', got '{freeze}'")

    return model


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def model_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6
