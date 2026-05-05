"""Recover secret models from stego checkpoints (lossless under POS / LSB SIH assumptions)."""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch.nn as nn

from core.gfi import remove_filter_at
from core.gfi_seq import conv_indices_in_seq
from core.sih import SideInfo, extract_payload_from_side, extract_payload_from_side_encoder
from models.vgg11 import VGG11
from core.gfi_seq import remove_filter_seq


def extract_secret_model(stego: VGG11, side: SideInfo, key: bytes, gfi_shapes: Dict[int, int]) -> VGG11:
    """
    gfi_shapes: rank -> number of channels per conv layer after GFI (before side), same as used for unpacking B.
    """
    masks = extract_payload_from_side(stego, side, key, gfi_shapes)
    m = copy.deepcopy(stego)
    m = remove_filter_at(m, side.conv_rank, side.channel_index)
    for rank in sorted(masks.keys(), reverse=True):
        bits = masks[rank]
        zeros = [i for i in range(bits.numel()) if int(bits[i].item()) == 0]
        for pos in sorted(zeros, reverse=True):
            m = remove_filter_at(m, rank, pos)
    return m


def extract_secret_encoder(
    stego: nn.Module,
    side: SideInfo,
    key: bytes,
    gfi_shapes: Dict[int, int],
    encoder_attr: str = "encoder",
) -> nn.Module:
    masks = extract_payload_from_side_encoder(stego, side, key, gfi_shapes, encoder_attr)
    m = copy.deepcopy(stego)
    enc = getattr(m, encoder_attr)
    enc = remove_filter_seq(enc, side.conv_rank, side.channel_index)
    setattr(m, encoder_attr, enc)
    for rank in sorted(masks.keys(), reverse=True):
        bits = masks[rank]
        zeros = [i for i in range(bits.numel()) if int(bits[i].item()) == 0]
        for pos in sorted(zeros, reverse=True):
            enc = getattr(m, encoder_attr)
            enc = remove_filter_seq(enc, rank, pos)
            setattr(m, encoder_attr, enc)
    return m


def relative_param_error(a: VGG11, b: VGG11) -> float:
    num = 0.0
    den = 0.0
    for (n1, p1), (n2, p2) in zip(a.named_parameters(), b.named_parameters()):
        assert n1 == n2
        d = (p1.detach() - p2.detach()).abs()
        num += d.sum().item()
        den += p2.detach().abs().sum().item()
    return num / max(den, 1e-12)


def relative_param_error_prefix(a: nn.Module, b: nn.Module, prefixes: Tuple[str, ...]) -> float:
    """Match parameters whose names start with any of `prefixes` (e.g. encoder + secret_head)."""
    dict_a = {n: p for n, p in a.named_parameters() if any(n.startswith(pref) for pref in prefixes)}
    dict_b = {n: p for n, p in b.named_parameters() if any(n.startswith(pref) for pref in prefixes)}
    num = 0.0
    den = 0.0
    for name in dict_a.keys():
        if name not in dict_b:
            continue
        p1, p2 = dict_a[name], dict_b[name]
        d = (p1.detach() - p2.detach()).abs()
        num += d.sum().item()
        den += p2.detach().abs().sum().item()
    return num / max(den, 1e-12)
