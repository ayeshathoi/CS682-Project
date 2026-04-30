#!/usr/bin/env python3
"""
Intra-task DNN steganography (Li et al., arXiv:2307.03444, Sec. 5.1):
  - Secret VGG11 trained on Fashion-MNIST (default split).
  - Stego model from secret + GFI + SIH, partial optimization on CIFAR-10 (D_st).
  - Metrics: ACC on both tasks; secret recoverability vs original checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from core.extract import extract_secret_model, relative_param_error
from core.gfi import (
    apply_insertions,
    compute_position_importance,
    count_total_insertion_positions,
    top_n_insertion_positions,
)
from core.masks import apply_grad_mask_, build_partial_masks, channel_roles_after_sih, statistical_loss_conv
from core.sih import insert_side_filter_and_embed
from data import cifar10_loaders, evaluate, fashion_mnist_loaders
from models.vgg11 import VGG11


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_classifier(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    desc: str,
) -> None:
    model.train()
    opt = Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        pbar = tqdm(train_loader, desc=f"{desc} ep{ep+1}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()


def train_stego_partial(
    stego: VGG11,
    grad_masks: dict,
    clean_ref: VGG11,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    alpha: float,
    beta: float,
    desc: str,
) -> None:
    stego.train()
    clean_ref.eval()
    opt = Adam(stego.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        pbar = tqdm(train_loader, desc=f"{desc} ep{ep+1}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = stego(x)
            l_st = crit(logits, y)
            l_mu, l_sig = statistical_loss_conv(stego, clean_ref)
            loss = l_st + alpha * l_mu + beta * l_sig
            loss.backward()
            apply_grad_mask_(stego, grad_masks)
            opt.step()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--epochs-secret", type=int, default=15)
    ap.add_argument("--epochs-clean", type=int, default=20)
    ap.add_argument("--epochs-stego", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=20.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--gfi-fraction", type=float, default=0.30, help="top fraction of insertion positions (paper: 0.30)")
    ap.add_argument("--gfi-max-batches", type=int, default=0, help="0 = full CIFAR train for GFI gradients; else cap batches")
    ap.add_argument("--key", type=str, default="cs682-secret-key-phrase!!")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    fm_train, fm_test = fashion_mnist_loaders(args.data_dir, args.batch_size)
    cf_train, cf_test = cifar10_loaders(args.data_dir, args.batch_size)

    secret = VGG11().to(device)
    print("Training secret VGG11 on Fashion-MNIST...")
    train_classifier(secret, fm_train, device, args.epochs_secret, args.lr, "secret FMNIST")
    acc_secret_fm = evaluate(secret, fm_test, device)
    print(f"Secret model Fashion-MNIST test ACC: {100.0 * acc_secret_fm:.2f}%")

    secret.eval()
    gfi_batches = None if args.gfi_max_batches == 0 else args.gfi_max_batches
    print("Computing GFI importance on CIFAR-10 (D_st)...")
    imp = compute_position_importance(secret, cf_train, device, max_batches=gfi_batches)
    total_pos = count_total_insertion_positions(secret)
    n_sel = max(1, int(round(args.gfi_fraction * total_pos)))
    specs = top_n_insertion_positions(imp, n_sel)
    print(f"Insertion slots: {total_pos}, selecting top {n_sel} (~{100.0 * n_sel / total_pos:.1f}%).")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed)
    stego_base, conv_masks = apply_insertions(secret, specs, rng)

    key = args.key.encode("utf-8")[:64]
    if len(key) < 8:
        key = key + b"\0" * (8 - len(key))

    stego_model, side_info = insert_side_filter_and_embed(stego_base, conv_masks, key, rng)
    stego_model = stego_model.to(device)

    gfi_shapes = {r: int(conv_masks[r].numel()) for r in sorted(conv_masks.keys())}
    roles = channel_roles_after_sih(conv_masks, side_info)
    grad_masks = build_partial_masks(stego_model, roles)

    clean = copy.deepcopy(stego_model)
    clean._initialize_weights()
    clean = clean.to(device)
    print("Training clean reference G_gamma on CIFAR-10 (full weights)...")
    train_classifier(clean, cf_train, device, args.epochs_clean, args.lr, "clean CIFAR")
    acc_clean = evaluate(clean, cf_test, device)
    print(f"Clean model CIFAR-10 test ACC: {100.0 * acc_clean:.2f}%")

    print("Training stego G_delta on CIFAR-10 (POS: interference only + stats losses)...")
    train_stego_partial(
        stego_model,
        grad_masks,
        clean,
        cf_train,
        device,
        args.epochs_stego,
        args.lr,
        args.alpha,
        args.beta,
        "stego CIFAR",
    )
    acc_stego = evaluate(stego_model, cf_test, device)
    print(f"Stego model CIFAR-10 test ACC: {100.0 * acc_stego:.2f}%")

    recovered = extract_secret_model(stego_model.cpu(), side_info, key, gfi_shapes).to(device)
    acc_rec_fm = evaluate(recovered, fm_test, device)
    print(f"Recovered secret Fashion-MNIST test ACC: {100.0 * acc_rec_fm:.2f}%")
    err = relative_param_error(secret.cpu(), recovered.cpu())
    print(f"Relative L1 param drift vs original secret: {err:.3e}")

    print("Done.")


if __name__ == "__main__":
    main()
