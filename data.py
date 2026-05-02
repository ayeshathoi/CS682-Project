"""Dataloaders: Fashion-MNIST (secret), CIFAR-10 (stego), Oxford-IIIT Pet seg (inter-task D_st)."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


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


_IMAGENET_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def oxford_pet_seg_loaders(
    data_dir: str,
    batch_size: int,
    img_size: int = 128,
    train_n: int = 6000,
    val_n: int = 1282,
    test_n: int = 100,
    seed: int = 0,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Oxford-IIIT Pet coarse segmentation trimap (foreground pet / border / background → 3 classes).
    Split sizes mirror Li et al. Sec. 5.1 inter-task setup when the trainval pool is large enough.
    """
    base = torchvision.datasets.OxfordIIITPet(
        data_dir,
        split="trainval",
        target_types=("segmentation",),
        download=True,
        transform=None,
        target_transform=None,
    )
    n_total = len(base)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total).tolist()
    need = train_n + val_n + test_n
    if need > n_total:
        train_n = int(n_total * 0.81)
        val_n = int(n_total * 0.17)
        test_n = n_total - train_n - val_n
        need = n_total
        perm = rng.permutation(n_total).tolist()

    train_idx = perm[:train_n]
    val_idx = perm[train_n : train_n + val_n]
    test_idx = perm[train_n + val_n : train_n + val_n + test_n]

    class OxfordPetSegSubset(torch.utils.data.Dataset):
        def __init__(self, indices: Sequence[int]):
            self.indices = list(indices)

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, i: int):
            idx = self.indices[i]
            img, mask = base[idx]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            if not isinstance(mask, Image.Image):
                mask = Image.fromarray(np.asarray(mask))
            img = img.convert("RGB")
            img = TF.resize(img, [img_size, img_size], antialias=True)
            mask = TF.resize(mask, [img_size, img_size], interpolation=InterpolationMode.NEAREST)
            x = TF.to_tensor(img)
            x = TF.normalize(x, _IMAGENET_NORM[0], _IMAGENET_NORM[1])
            mt = TF.pil_to_tensor(mask).long().squeeze(0)
            mt = mt - 1
            mt = torch.clamp(mt, 0, 2)
            return x, mt

    train_ds = OxfordPetSegSubset(train_idx)
    val_ds = OxfordPetSegSubset(val_idx)
    test_ds = OxfordPetSegSubset(test_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, 3


@torch.no_grad()
def mean_iou(model: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> float:
    model.eval()
    inter = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model.forward_stego(x)
        pred = logits.argmax(dim=1)
        for c in range(num_classes):
            pb = pred == c
            yb = y == c
            inter[c] += (pb & yb).sum().float()
            union[c] += (pb | yb).sum().float()
    iou = inter / union.clamp(min=1.0)
    mask_cls = union > 0
    if mask_cls.any():
        return iou[mask_cls].mean().item()
    return 0.0


@torch.no_grad()
def psnr_denoise_normalized(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """PSNR assuming pixel values lie in roughly [-1, 1] after normalization (peak amplitude 2)."""
    mse = (pred - target).pow(2).mean(dim=(1, 2, 3))
    mse = torch.clamp(mse, min=eps)
    psnr = 10.0 * torch.log10((torch.tensor(4.0, device=mse.device, dtype=mse.dtype) / mse))
    return float(psnr.mean().item())
