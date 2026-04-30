"""Recover secret VGG11 from stego model (lossless in exact arithmetic / LSB fidelity)."""

from __future__ import annotations

import copy
from typing import Dict

import torch

from core.gfi import remove_filter_at
from core.sih import SideInfo, extract_payload_from_side
from models.vgg11 import VGG11


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


def relative_param_error(a: VGG11, b: VGG11) -> float:
    num = 0.0
    den = 0.0
    for (n1, p1), (n2, p2) in zip(a.named_parameters(), b.named_parameters()):
        assert n1 == n2
        d = (p1.detach() - p2.detach()).abs()
        num += d.sum().item()
        den += p2.detach().abs().sum().item()
    return num / max(den, 1e-12)
