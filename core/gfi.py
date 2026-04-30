"""Gradient-based filter insertion (GFI) for VGG11."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from models.vgg11 import VGG11


@dataclass
class InsertionSpec:
    conv_rank: int  # which conv layer 0..num_conv-1 (in order of appearance)
    position: int  # insert before this filter index, 0..out_channels (inclusive)


def _features_indices_of_convs(features: nn.Sequential) -> List[int]:
    return [i for i, m in enumerate(features) if isinstance(m, nn.Conv2d)]


def _iter_cifar_batches(loader, device, max_batches: Optional[int] = None):
    n = 0
    for x, y in loader:
        yield x.to(device), y.to(device)
        n += 1
        if max_batches is not None and n >= max_batches:
            break


def compute_position_importance(
    model: VGG11,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[Tuple[int, int], float]:
    """
    Eq. (2)-(4): gradient magnitude summed over D_st, then per-filter mean,
    then insertion position importance between neighbors.
    Returns map (conv_rank, position_j) -> p^l_j for j in 1..d+1 (positions 0..d in 0-based between filters).
    """
    model = copy.deepcopy(model).to(device)
    model.train()
    conv_idx_list = _features_indices_of_convs(model.features)

    # Accumulate |grad| per conv weight tensor (same shape as weight)
    grad_sum: Dict[int, torch.Tensor] = {}
    for fi in conv_idx_list:
        grad_sum[fi] = None

    criterion = nn.CrossEntropyLoss()
    for xb, yb in _iter_cifar_batches(data_loader, device, max_batches):
        model.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        for fi in conv_idx_list:
            conv = model.features[fi]
            assert isinstance(conv, nn.Conv2d)
            g = conv.weight.grad
            if g is None:
                continue
            acc = torch.abs(g.detach())
            if grad_sum[fi] is None:
                grad_sum[fi] = acc.clone()
            else:
                grad_sum[fi] += acc

    importance: Dict[Tuple[int, int], float] = {}
    for rank, fi in enumerate(conv_idx_list):
        acc = grad_sum[fi]
        if acc is None:
            acc = torch.zeros_like(model.features[fi].weight)
        d = acc.shape[0]
        # w_i = mean over (c, k, k) for each filter i
        w = acc.view(d, -1).mean(dim=1)  # (d,)
        # positions j=1..d+1 -> 0-based position index for "slot" before filter j
        for j in range(d + 1):
            if j == 0:
                p = float(w[0].item())
            elif j == d:
                p = float(w[d - 1].item())
            else:
                p = 0.5 * (float(w[j - 1].item()) + float(w[j].item()))
            importance[(rank, j)] = p
    return importance


def top_n_insertion_positions(
    importance: Dict[Tuple[int, int], float], n_select: int
) -> List[InsertionSpec]:
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    specs: List[InsertionSpec] = []
    for (rank, j), _ in items[:n_select]:
        specs.append(InsertionSpec(conv_rank=rank, position=j))
    return specs


def _next_conv_features_index(features: nn.Sequential, after_idx: int) -> Optional[int]:
    for i in range(after_idx + 1, len(features)):
        if isinstance(features[i], nn.Conv2d):
            return i
    return None


def insert_one_filter(
    model: VGG11,
    conv_rank: int,
    position: int,
    rng: torch.Generator,
) -> Tuple[VGG11, int]:
    """
    Insert one interference filter at given conv_rank and position (0..out_ch).
    Position is interpreted in the **current** model's filter indexing for that layer.
    Returns updated model and the features index of the modified conv.
    """
    device = next(model.parameters()).device
    features = model.features
    conv_indices = _features_indices_of_convs(features)
    fi = conv_indices[conv_rank]
    conv = features[fi]
    assert isinstance(conv, nn.Conv2d)
    in_ch, out_ch, kh, kw = conv.in_channels, conv.out_channels, conv.kernel_size[0], conv.kernel_size[1]

    new_conv = nn.Conv2d(
        in_ch, out_ch + 1, kernel_size=kh, padding=conv.padding[0], bias=conv.bias is not None
    ).to(device)
    w_old = conv.weight.data
    b_old = conv.bias.data if conv.bias is not None else None
    w_new = new_conv.weight.data
    b_new = new_conv.bias.data if new_conv.bias is not None else None

    # copy rows with insertion at `position`
    if position > 0:
        w_new[:position].copy_(w_old[:position])
    if position < out_ch:
        w_new[position + 1 :].copy_(w_old[position:])
    nn.init.normal_(w_new[position : position + 1], 0, 0.02)
    if b_new is not None:
        if position > 0:
            b_new[:position].copy_(b_old[:position])
        if position < out_ch:
            b_new[position + 1 :].copy_(b_old[position:])
        b_new[position].zero_()

    new_features_modules = []
    for i, m in enumerate(features):
        if i == fi:
            new_features_modules.append(new_conv)
        else:
            new_features_modules.append(m)
    new_features = nn.Sequential(*new_features_modules)

    next_fi = _next_conv_features_index(new_features, fi)
    if next_fi is not None:
        nconv = new_features[next_fi]
        assert isinstance(nconv, nn.Conv2d)
        # Next layer's input depth must match this layer's output depth (after insertion).
        new_next = nn.Conv2d(
            in_channels=out_ch + 1,
            out_channels=nconv.out_channels,
            kernel_size=nconv.kernel_size[0],
            padding=nconv.padding[0],
            bias=nconv.bias is not None,
        ).to(device)
        wn_old = nconv.weight.data
        wn_new = new_next.weight.data
        # expand input channel dimension at index `position`
        if position > 0:
            wn_new[:, :position].copy_(wn_old[:, :position])
        nn.init.normal_(wn_new[:, position : position + 1], 0, 0.02)
        if position < out_ch:
            wn_new[:, position + 1 :].copy_(wn_old[:, position:])
        if new_next.bias is not None and nconv.bias is not None:
            new_next.bias.data.copy_(nconv.bias.data)
        nf_mods = []
        for j, m in enumerate(new_features):
            if j == next_fi:
                nf_mods.append(new_next)
            else:
                nf_mods.append(m)
        new_features = nn.Sequential(*nf_mods)
    else:
        # Last conv: attach to classifier input dim
        old_lin = model.classifier
        assert isinstance(old_lin, nn.Linear)
        new_lin = nn.Linear(old_lin.in_features + 1, old_lin.out_features).to(device)
        w_lin = old_lin.weight.data
        b_lin = old_lin.bias.data
        wl_new = new_lin.weight.data
        if position > 0:
            wl_new[:, :position].copy_(w_lin[:, :position])
        nn.init.normal_(wl_new[:, position : position + 1], 0, 0.02)
        if position < old_lin.in_features:
            wl_new[:, position + 1 :].copy_(w_lin[:, position:])
        new_lin.bias.data.copy_(b_lin)
        model.classifier = new_lin

    model.features = new_features
    return model, fi


def _final_interference_indices_for_layer(original_positions_desc: List[int]) -> List[int]:
    """original_positions_desc: insertion positions processed in descending order."""
    final_set = set()
    for p in original_positions_desc:
        final_set = {i + 1 if i >= p else i for i in final_set}
        final_set.add(p)
    return sorted(final_set)


def apply_insertions(
    model: VGG11,
    specs: Sequence[InsertionSpec],
    rng: torch.Generator,
) -> Tuple[VGG11, Dict[int, torch.Tensor]]:
    """
    Apply insertions. Within each conv_rank, process positions in descending order
    so each position refers to the secret model's indexing.
    Returns model and per-conv-rank uint8 vector (1=original, 0=interference).
    """
    by_rank: Dict[int, List[int]] = {}
    for s in specs:
        by_rank.setdefault(s.conv_rank, []).append(s.position)
    ordered: List[InsertionSpec] = []
    for rank in sorted(by_rank.keys()):
        for pos in sorted(by_rank[rank], reverse=True):
            ordered.append(InsertionSpec(conv_rank=rank, position=pos))

    m = copy.deepcopy(model)
    rank_to_final_indices: Dict[int, List[int]] = {}
    for rank in by_rank:
        rank_to_final_indices[rank] = _final_interference_indices_for_layer(
            sorted(by_rank[rank], reverse=True)
        )

    for s in ordered:
        m, _ = insert_one_filter(m, s.conv_rank, s.position, rng)

    conv_masks: Dict[int, torch.Tensor] = {}
    conv_indices = _features_indices_of_convs(m.features)
    for rank, fi in enumerate(conv_indices):
        conv = m.features[fi]
        assert isinstance(conv, nn.Conv2d)
        d = conv.out_channels
        bits = torch.ones(d, dtype=torch.uint8)
        for idx in rank_to_final_indices.get(rank, []):
            if 0 <= idx < d:
                bits[idx] = 0
        conv_masks[rank] = bits
    return m, conv_masks


def remove_filter_at(model: VGG11, conv_rank: int, position: int) -> VGG11:
    """Inverse of insert_one_filter: remove output channel at `position`."""
    device = next(model.parameters()).device
    features = model.features
    conv_indices = _features_indices_of_convs(features)
    fi = conv_indices[conv_rank]
    conv = features[fi]
    assert isinstance(conv, nn.Conv2d)
    in_ch, out_ch, kh, kw = conv.in_channels, conv.out_channels, conv.kernel_size[0], conv.kernel_size[1]
    assert 0 <= position < out_ch
    new_conv = nn.Conv2d(
        in_ch, out_ch - 1, kernel_size=kh, padding=conv.padding[0], bias=conv.bias is not None
    ).to(device)
    w_old = conv.weight.data
    b_old = conv.bias.data if conv.bias is not None else None
    w_new = new_conv.weight.data
    b_new = new_conv.bias.data if new_conv.bias is not None else None
    if position > 0:
        w_new[:position].copy_(w_old[:position])
    if position + 1 < out_ch:
        w_new[position:].copy_(w_old[position + 1 :])
    if b_new is not None and b_old is not None:
        if position > 0:
            b_new[:position].copy_(b_old[:position])
        if position + 1 < out_ch:
            b_new[position:].copy_(b_old[position + 1 :])

    new_features_modules = []
    for i, m in enumerate(features):
        if i == fi:
            new_features_modules.append(new_conv)
        else:
            new_features_modules.append(m)
    new_features = nn.Sequential(*new_features_modules)

    next_fi = _next_conv_features_index(new_features, fi)
    if next_fi is not None:
        nconv = new_features[next_fi]
        assert isinstance(nconv, nn.Conv2d)
        new_next = nn.Conv2d(
            in_channels=out_ch - 1,
            out_channels=nconv.out_channels,
            kernel_size=nconv.kernel_size[0],
            padding=nconv.padding[0],
            bias=nconv.bias is not None,
        ).to(device)
        wn_old = nconv.weight.data
        wn_new = new_next.weight.data
        if position > 0:
            wn_new[:, :position].copy_(wn_old[:, :position])
        if position + 1 < out_ch:
            wn_new[:, position:].copy_(wn_old[:, position + 1 :])
        if new_next.bias is not None and nconv.bias is not None:
            new_next.bias.data.copy_(nconv.bias.data)
        nf_mods = []
        for j, m in enumerate(new_features):
            if j == next_fi:
                nf_mods.append(new_next)
            else:
                nf_mods.append(m)
        new_features = nn.Sequential(*nf_mods)
    else:
        old_lin = model.classifier
        assert isinstance(old_lin, nn.Linear)
        new_lin = nn.Linear(old_lin.in_features - 1, old_lin.out_features).to(device)
        w_lin = old_lin.weight.data
        b_lin = old_lin.bias.data
        wl_new = new_lin.weight.data
        if position > 0:
            wl_new[:, :position].copy_(w_lin[:, :position])
        if position + 1 < old_lin.in_features:
            wl_new[:, position:].copy_(w_lin[:, position + 1 :])
        new_lin.bias.data.copy_(b_lin)
        model.classifier = new_lin

    model.features = new_features
    return model


def count_total_insertion_positions(model: VGG11) -> int:
    conv_indices = _features_indices_of_convs(model.features)
    total = 0
    for fi in conv_indices:
        conv = model.features[fi]
        assert isinstance(conv, nn.Conv2d)
        total += conv.out_channels + 1
    return total
