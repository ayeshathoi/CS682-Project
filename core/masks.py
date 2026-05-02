"""Partial optimization masks and statistical loss helpers."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from core.gfi import _features_indices_of_convs
from core.gfi_seq import conv_indices_in_seq
from core.sih import SideInfo
from models.vgg11 import VGG11

CH_INT = 0  # interference (train)
CH_ORIG = 1  # original secret weights (frozen)
CH_SIDE = 2  # side filter (frozen)


def channel_roles_after_sih(conv_masks: Dict[int, torch.Tensor], side: SideInfo) -> Dict[int, torch.Tensor]:
    """0=interference, 1=original, 2=side. Length matches channels in each conv after SIH."""
    out: Dict[int, torch.Tensor] = {}
    pos = side.channel_index
    for rank, bits in sorted(conv_masks.items()):
        if rank != side.conv_rank:
            out[rank] = bits.to(torch.long).clone()
            continue
        d = bits.numel()
        row = torch.zeros(d + 1, dtype=torch.long, device=bits.device)
        row[:pos] = bits[:pos].to(torch.long)
        row[pos] = CH_SIDE
        if pos < d:
            row[pos + 1 :] = bits[pos:].to(torch.long)
        out[rank] = row
    return out


def conv_weight_bias_masks(
    conv: nn.Conv2d,
    out_roles: torch.Tensor,
    in_roles: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    out_c, in_c = conv.out_channels, conv.in_channels
    w_mask = torch.zeros(out_c, in_c, 1, 1, device=conv.weight.device, dtype=conv.weight.dtype)
    for o in range(out_c):
        if out_roles[o].item() != CH_INT:
            continue
        for i in range(in_c):
            if in_roles is None:
                w_mask[o, i, 0, 0] = 1.0
                continue
            if in_roles[i].item() == CH_SIDE:
                continue
            w_mask[o, i, 0, 0] = 1.0
    b_mask = torch.zeros(out_c, device=conv.weight.device, dtype=conv.weight.dtype)
    for o in range(out_c):
        if out_roles[o].item() == CH_INT:
            b_mask[o] = 1.0
    return w_mask, b_mask


def linear_masks(lin: nn.Linear, in_roles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    w = torch.zeros_like(lin.weight)
    o_dim, i_dim = lin.weight.shape
    for o in range(o_dim):
        for i in range(i_dim):
            if in_roles[i].item() == CH_INT:
                w[o, i] = 1.0
    b = torch.ones_like(lin.bias)
    return w, b


def build_partial_masks(model: VGG11, roles: Dict[int, torch.Tensor]) -> Dict[str, torch.Tensor]:
    conv_idx = _features_indices_of_convs(model.features)
    per_rank_in_roles: Dict[int, Optional[torch.Tensor]] = {}
    for rank in sorted(roles.keys()):
        if rank == 0:
            per_rank_in_roles[rank] = None
        else:
            per_rank_in_roles[rank] = roles[rank - 1]
    idx_by_rank = {r: conv_idx[r] for r in range(len(conv_idx))}

    param_to_mask: Dict[str, torch.Tensor] = {}
    for rank, fi in enumerate(conv_idx):
        conv = model.features[fi]
        assert isinstance(conv, nn.Conv2d)
        out_roles = roles[rank]
        in_roles = per_rank_in_roles[rank]
        wm, bm = conv_weight_bias_masks(conv, out_roles, in_roles)
        wm_full = wm.expand_as(conv.weight)
        param_to_mask[f"features.{fi}.weight"] = wm_full
        if conv.bias is not None:
            param_to_mask[f"features.{fi}.bias"] = bm

    lin = model.classifier
    assert isinstance(lin, nn.Linear)
    last_rank = len(conv_idx) - 1
    w_lin_m, b_lin_m = linear_masks(lin, roles[last_rank])
    param_to_mask["classifier.weight"] = w_lin_m
    param_to_mask["classifier.bias"] = b_lin_m
    return param_to_mask


def apply_grad_mask_(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        m = masks.get(name)
        if m is None:
            p.grad.zero_()
            continue
        p.grad.mul_(m)


def statistical_loss_conv(model_a: VGG11, model_b: VGG11) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. (6): L_mu and L_sigma over convolutional layers only."""
    idx_a = _features_indices_of_convs(model_a.features)
    idx_b = _features_indices_of_convs(model_b.features)
    assert idx_a == idx_b
    l_mu = model_a.features[0].weight.new_tensor(0.0)
    l_sig = model_a.features[0].weight.new_tensor(0.0)
    for fi in idx_a:
        wa = model_a.features[fi].weight
        wb = model_b.features[fi].weight
        ma, sa = wa.mean(), wa.std()
        mb, sb = wb.mean(), wb.std()
        l_mu = l_mu + (ma - mb).pow(2)
        l_sig = l_sig + (sa - sb).pow(2)
    return l_mu, l_sig


def build_partial_masks_dual_encoder(
    model: nn.Module,
    roles: Dict[int, torch.Tensor],
    encoder_attr: str = "encoder",
) -> Dict[str, torch.Tensor]:
    """POS masks for DualHeadDnCNN-style modules: train interference + stego head; freeze secret + side."""
    enc = getattr(model, encoder_attr)
    conv_idx = conv_indices_in_seq(enc)
    per_rank_in_roles: Dict[int, Optional[torch.Tensor]] = {}
    for rank in sorted(roles.keys()):
        per_rank_in_roles[rank] = None if rank == 0 else roles[rank - 1]

    param_to_mask: Dict[str, torch.Tensor] = {}
    for rank, fi in enumerate(conv_idx):
        conv = enc[fi]
        assert isinstance(conv, nn.Conv2d)
        wm, bm = conv_weight_bias_masks(conv, roles[rank], per_rank_in_roles[rank])
        param_to_mask[f"{encoder_attr}.{fi}.weight"] = wm.expand_as(conv.weight)
        if conv.bias is not None:
            param_to_mask[f"{encoder_attr}.{fi}.bias"] = bm

    sh = model.secret_head
    param_to_mask["secret_head.weight"] = torch.zeros_like(sh.weight)
    th = model.stego_head
    param_to_mask["stego_head.weight"] = torch.ones_like(th.weight)
    if th.bias is not None:
        param_to_mask["stego_head.bias"] = torch.ones_like(th.bias)
    return param_to_mask


def statistical_loss_encoder(
    model_a: nn.Module,
    model_b: nn.Module,
    encoder_attr: str = "encoder",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. (6) over Conv2d weights inside encoder Sequential."""
    seq_a = getattr(model_a, encoder_attr)
    seq_b = getattr(model_b, encoder_attr)
    idx_a = conv_indices_in_seq(seq_a)
    idx_b = conv_indices_in_seq(seq_b)
    assert idx_a == idx_b
    l_mu = seq_a[idx_a[0]].weight.new_tensor(0.0)
    l_sig = seq_a[idx_a[0]].weight.new_tensor(0.0)
    for fi in idx_a:
        wa = seq_a[fi].weight
        wb = seq_b[fi].weight
        ma, sa = wa.mean(), wa.std()
        mb, sb = wb.mean(), wb.std()
        l_mu = l_mu + (ma - mb).pow(2)
        l_sig = l_sig + (sa - sb).pow(2)
    return l_mu, l_sig
