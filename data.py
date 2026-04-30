"""Dataloaders: Fashion-MNIST (secret) and CIFAR-10 (stego, D_st)."""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


def fashion_mnist_loaders(
    data_dir: str, batch_size: int, num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    tfm = T.Compose(
        [
            T.Resize(32),
            T.ToTensor(),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=tfm)
    test = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=tfm)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )


def cifar10_loaders(
    data_dir: str, batch_size: int, num_workers: int = 2
) -> Tuple[DataLoader, DataLoader]:
    tfm = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )
    train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=tfm)
    test = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=tfm)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True),
    )


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)
