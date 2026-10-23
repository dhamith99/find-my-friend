"""Load the .pt tensor files and apply augmentation (train only)."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

# ImageNet statistics - the pretrained models expect inputs normalised like this
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def train_transform(size: int):
    return v2.Compose([
        v2.RandomResizedCrop(size, scale=(0.8, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(10),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN, STD),
    ])


def eval_transform():
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN, STD),
    ])


def load_split(path):
    return torch.load(path, weights_only=True)


class FaceDataset(Dataset):
    def __init__(self, path, train: bool = False):
        d = load_split(path)
        self.images = d["images"]          # uint8 [N, 3, S, S]
        self.labels = d["labels"]
        self.classes = d["classes"]
        size = self.images.shape[-1]
        self.tf = train_transform(size) if train else eval_transform()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.tf(self.images[i]), self.labels[i]
