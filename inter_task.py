#!/usr/bin/env python3
"""
Inter-task DNN steganography (Li et al., arXiv:2307.03444, Sec. 5.1):
  Secret: DualHeadDnCNN trained for Gaussian denoising on Oxford-Pet RGB images.
  Stego D_st: Oxford-IIIT Pet trimap segmentation (3 classes).
  Pipeline: GFI (gradient importance on segmentation loss) → SIH → POS with L_st + α L_μ + β L_σ.

Insertion slots exclude the **last encoder conv rank** so encoder output width stays `width`
and matches the frozen secret_head shape after extraction (same rationale as constraining POS scope).

Capabilities demonstrated:
  - Sender embeds a functional denoising CNN inside a segmentation-shaped public model (receiver +
    key recover denoiser weights losslessly).
  - Metrics: segmentation mIoU (stego vs clean reference), PSNR on noisy→denoise test batches,
    relative parameter drift on encoder+secret_head vs original secret checkpoint.
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
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

from core.extract import extract_secret_encoder, relative_param_error_prefix
from core.gfi_seq import (
    apply_insertions_seq,
    compute_position_importance_seq,
    count_total_insertion_positions_seq,
    conv_indices_in_seq,
    top_n_positions,
)
from core.masks import (
    apply_grad_mask_,
    build_partial_masks_dual_encoder,
    channel_roles_after_sih,
    statistical_loss_encoder,
)
from core.sih import insert_side_filter_and_embed_encoder
from data import mean_iou, oxford_pet_seg_loaders, psnr_denoise_normalized
from models.dncnn_dual import DualHeadDnCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_denoiser(
    model: DualHeadDnCNN,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    noise_std: float,
    desc: str,
) -> None:
    model.train()
    opt = Adam(
        [
            {"params": model.encoder.parameters()},
            {"params": model.secret_head.parameters()},
        ],
        lr=lr,
    )
    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"{desc} ep{ep+1}/{epochs}", leave=False)
        for x, _ in pbar:
            x = x.to(device)
            noise = torch.randn_like(x) * noise_std
            noisy = x + noise
            opt.zero_grad(set_to_none=True)
            pred = model.forward_secret(noisy)
            loss = F.l1_loss(pred, x)
            loss.backward()
            opt.step()


def train_seg_full(model: DualHeadDnCNN, loader: torch.utils.data.DataLoader, device: torch.device, epochs: int, lr: float, desc: str) -> None:
    model.train()
    opt = Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()
    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"{desc} ep{ep+1}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model.forward_stego(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()


def train_stego_partial(
    stego: DualHeadDnCNN,
    grad_masks: dict,
    clean_ref: DualHeadDnCNN,
    loader: torch.utils.data.DataLoader,
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
    crit = torch.nn.CrossEntropyLoss()
    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"{desc} ep{ep+1}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = stego.forward_stego(x)
            l_st = crit(logits, y)
            l_mu, l_sig = statistical_loss_encoder(stego, clean_ref, "encoder")
            loss = l_st + alpha * l_mu + beta * l_sig
            loss.backward()
            apply_grad_mask_(stego, grad_masks)
            opt.step()


@torch.no_grad()
def average_psnr_denoise(model: DualHeadDnCNN, loader: torch.utils.data.DataLoader, device: torch.device, noise_std: float) -> float:
    model.eval()
    tot = 0.0
    n = 0
    for x, _ in loader:
        x = x.to(device)
        noise = torch.randn_like(x) * noise_std
        noisy = x + noise
        pred = model.forward_secret(noisy)
        tot += psnr_denoise_normalized(pred, x) * x.shape[0]
        n += x.shape[0]
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=str(ROOT / "data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--epochs-secret", type=int, default=8)
    ap.add_argument("--epochs-clean", type=int, default=12)
    ap.add_argument("--epochs-stego", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=20.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--noise-std", type=float, default=0.18, help="Gaussian noise scale on normalized RGB for secret denoise training/eval.")
    ap.add_argument("--gfi-fraction", type=float, default=0.30)
    ap.add_argument("--gfi-max-batches", type=int, default=40, help="Cap gradient batches for GFI (0 = full train loader).")
    ap.add_argument("--encoder-convs", type=int, default=8)
    ap.add_argument("--encoder-width", type=int, default=64)
    ap.add_argument("--key", type=str, default="cs682-inter-task-secret-key!")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    pet_train, _pet_val, pet_test, num_seg_classes = oxford_pet_seg_loaders(
        args.data_dir,
        args.batch_size,
        img_size=args.img_size,
        seed=args.seed,
    )

    secret = DualHeadDnCNN(
        num_seg_classes=num_seg_classes,
        num_encoder_convs=args.encoder_convs,
        width=args.encoder_width,
        init_weights=True,
    ).to(device)

    print("Training secret denoiser (encoder + secret_head) on Oxford-Pet train RGB...")
    train_denoiser(secret, pet_train, device, args.epochs_secret, args.lr, args.noise_std, "secret denoise")
    ps_before = average_psnr_denoise(secret, pet_test, device, args.noise_std)
    print(f"Secret denoiser held-out PSNR (normalized-space metric): {ps_before:.2f} dB")

    probe = copy.deepcopy(secret).to(device)
    crit = torch.nn.CrossEntropyLoss()

    def loss_forward_seg(m: DualHeadDnCNN, xb: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        return crit(m.forward_stego(xb), yb)

    max_b = None if args.gfi_max_batches == 0 else args.gfi_max_batches
    print("Computing GFI insertion importance using segmentation probe + D_st...")
    imp = compute_position_importance_seq(
        probe,
        "encoder",
        pet_train,
        device,
        loss_forward_seg,
        max_batches=max_b,
    )
    total_pos = count_total_insertion_positions_seq(secret.encoder)
    n_sel = max(1, int(round(args.gfi_fraction * total_pos)))
    specs_all = top_n_positions(imp, n_sel)
    n_cr = len(conv_indices_in_seq(secret.encoder))
    specs = [s for s in specs_all if s.conv_rank < n_cr - 1]
    print(f"Insertion slots: {total_pos}; selecting top-{n_sel}, keeping {len(specs)} after excluding last encoder conv.")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(args.seed)

    stego_base = copy.deepcopy(secret)
    new_enc, conv_masks = apply_insertions_seq(stego_base.encoder, specs, rng)
    stego_base.encoder = new_enc

    key = args.key.encode("utf-8")[:64]
    if len(key) < 8:
        key = key + b"\0" * (8 - len(key))

    stego_model, side_info = insert_side_filter_and_embed_encoder(stego_base, "encoder", conv_masks, key, rng)
    stego_model = stego_model.to(device)

    gfi_shapes = {r: int(conv_masks[r].numel()) for r in sorted(conv_masks.keys())}
    roles = channel_roles_after_sih(conv_masks, side_info)
    grad_masks = build_partial_masks_dual_encoder(stego_model, roles, "encoder")

    clean = copy.deepcopy(stego_model)
    clean._initialize_weights()
    clean = clean.to(device)
    print("Training clean reference G_gamma (full weights, segmentation)...")
    train_seg_full(clean, pet_train, device, args.epochs_clean, args.lr, "clean seg")
    mi_clean = mean_iou(clean, pet_test, device, num_seg_classes)
    print(f"Clean reference test mIoU: {100.0 * mi_clean:.2f}%")

    print("Training stego G_delta (POS: interference + stego_head + stats losses)...")
    train_stego_partial(
        stego_model,
        grad_masks,
        clean,
        pet_train,
        device,
        args.epochs_stego,
        args.lr,
        args.alpha,
        args.beta,
        "stego seg",
    )
    mi_stego = mean_iou(stego_model, pet_test, device, num_seg_classes)
    print(f"Stego model test mIoU: {100.0 * mi_stego:.2f}%")

    recovered = extract_secret_encoder(stego_model.cpu(), side_info, key, gfi_shapes).to(device)
    ps_rec = average_psnr_denoise(recovered, pet_test, device, args.noise_std)
    print(f"Recovered secret PSNR on same noisy→clean benchmark: {ps_rec:.2f} dB")

    drift = relative_param_error_prefix(secret.cpu(), recovered.cpu(), ("encoder.", "secret_head."))
    print(f"Relative L1 drift encoder+secret_head vs original secret: {drift:.3e}")

    print("Done.")


if __name__ == "__main__":
    main()
