"""Gradient-based filter insertion on arbitrary nn.Sequential stacks (Conv2d only indexed)."""

from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from core.gfi import InsertionSpec


def conv_indices_in_seq(seq: nn.Sequential) -> List[int]:
    return [i for i, m in enumerate(seq) if isinstance(m, nn.Conv2d)]


def _next_conv_index(seq: nn.Sequential, after_idx: int) -> Optional[int]:
    for i in range(after_idx + 1, len(seq)):
        if isinstance(seq[i], nn.Conv2d):
            return i
    return None


def insert_one_filter_seq(
    seq: nn.Sequential,
    conv_rank: int,
    position: int,
    rng: torch.Generator,
) -> nn.Sequential:
    """
    Insert one interference filter at conv_rank / position (0..out_ch).
    Does not assume a trailing Linear classifier — if no next Conv2d exists,
    only the current conv expands along output filters.
    """
    conv_ix_list = conv_indices_in_seq(seq)
    fi = conv_ix_list[conv_rank]
    conv = seq[fi]
    assert isinstance(conv, nn.Conv2d)
    device = conv.weight.device
    dtype = conv.weight.dtype
    in_ch, out_ch, kh, kw = conv.in_channels, conv.out_channels, conv.kernel_size[0], conv.kernel_size[1]

    new_conv = nn.Conv2d(
        in_ch, out_ch + 1, kernel_size=kh, padding=conv.padding[0], bias=conv.bias is not None
    ).to(device=device, dtype=dtype)
    w_old = conv.weight.data
    b_old = conv.bias.data if conv.bias is not None else None
    w_new = new_conv.weight.data
    b_new = new_conv.bias.data if new_conv.bias is not None else None

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

    mods = list(seq.children())
    mods[fi] = new_conv
    new_seq = nn.Sequential(*mods)

    next_fi = _next_conv_index(new_seq, fi)
    if next_fi is not None:
        nconv = new_seq[next_fi]
        assert isinstance(nconv, nn.Conv2d)
        new_next = nn.Conv2d(
            in_channels=out_ch + 1,
            out_channels=nconv.out_channels,
            kernel_size=nconv.kernel_size[0],
            padding=nconv.padding[0],
            bias=nconv.bias is not None,
        ).to(device=device, dtype=dtype)
        wn_old = nconv.weight.data
        wn_new = new_next.weight.data
        if position > 0:
            wn_new[:, :position].copy_(wn_old[:, :position])
        nn.init.normal_(wn_new[:, position : position + 1], 0, 0.02)
        if position < out_ch:
            wn_new[:, position + 1 :].copy_(wn_old[:, position:])
        if new_next.bias is not None and nconv.bias is not None:
            new_next.bias.data.copy_(nconv.bias.data)
        mods2 = list(new_seq.children())
        mods2[next_fi] = new_next
        new_seq = nn.Sequential(*mods2)

    return new_seq


def remove_filter_seq(seq: nn.Sequential, conv_rank: int, position: int) -> nn.Sequential:
    conv_ix_list = conv_indices_in_seq(seq)
    fi = conv_ix_list[conv_rank]
    conv = seq[fi]
    assert isinstance(conv, nn.Conv2d)
    device = conv.weight.device
    dtype = conv.weight.dtype
    in_ch, out_ch, kh, kw = conv.in_channels, conv.out_channels, conv.kernel_size[0], conv.kernel_size[1]
    assert 0 <= position < out_ch

    new_conv = nn.Conv2d(
        in_ch, out_ch - 1, kernel_size=kh, padding=conv.padding[0], bias=conv.bias is not None
    ).to(device=device, dtype=dtype)
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

    mods = list(seq.children())
    mods[fi] = new_conv
    new_seq = nn.Sequential(*mods)

    next_fi = _next_conv_index(new_seq, fi)
    if next_fi is not None:
        nconv = new_seq[next_fi]
        assert isinstance(nconv, nn.Conv2d)
        new_next = nn.Conv2d(
            in_channels=out_ch - 1,
            out_channels=nconv.out_channels,
            kernel_size=nconv.kernel_size[0],
            padding=nconv.padding[0],
            bias=nconv.bias is not None,
        ).to(device=device, dtype=dtype)
        wn_old = nconv.weight.data
        wn_new = new_next.weight.data
        if position > 0:
            wn_new[:, :position].copy_(wn_old[:, :position])
        if position + 1 < out_ch:
            wn_new[:, position:].copy_(wn_old[:, position + 1 :])
        if new_next.bias is not None and nconv.bias is not None:
            new_next.bias.data.copy_(nconv.bias.data)
        mods2 = list(new_seq.children())
        mods2[next_fi] = new_next
        new_seq = nn.Sequential(*mods2)

    return new_seq


def _final_interference_indices(original_positions_desc: List[int]) -> List[int]:
    final_set = set()
    for p in original_positions_desc:
        final_set = {i + 1 if i >= p else i for i in final_set}
        final_set.add(p)
    return sorted(final_set)


def apply_insertions_seq(
    encoder: nn.Sequential,
    specs: Sequence[InsertionSpec],
    rng: torch.Generator,
) -> Tuple[nn.Sequential, Dict[int, torch.Tensor]]:
    by_rank: Dict[int, List[int]] = {}
    for s in specs:
        by_rank.setdefault(s.conv_rank, []).append(s.position)
    ordered: List[InsertionSpec] = []
    for rank in sorted(by_rank.keys()):
        for pos in sorted(by_rank[rank], reverse=True):
            ordered.append(InsertionSpec(conv_rank=rank, position=pos))

    rank_to_final_indices: Dict[int, List[int]] = {}
    for rank in by_rank:
        rank_to_final_indices[rank] = _final_interference_indices(sorted(by_rank[rank], reverse=True))

    seq = encoder
    for s in ordered:
        seq = insert_one_filter_seq(seq, s.conv_rank, s.position, rng)

    conv_masks: Dict[int, torch.Tensor] = {}
    conv_ix_list = conv_indices_in_seq(seq)
    for rank, fi in enumerate(conv_ix_list):
        conv = seq[fi]
        assert isinstance(conv, nn.Conv2d)
        d = conv.out_channels
        bits = torch.ones(d, dtype=torch.uint8)
        for idx in rank_to_final_indices.get(rank, []):
            if 0 <= idx < d:
                bits[idx] = 0
        conv_masks[rank] = bits
    return seq, conv_masks


def count_total_insertion_positions_seq(encoder: nn.Sequential) -> int:
    total = 0
    for fi in conv_indices_in_seq(encoder):
        conv = encoder[fi]
        assert isinstance(conv, nn.Conv2d)
        total += conv.out_channels + 1
    return total


def compute_position_importance_seq(
    model: nn.Module,
    encoder_attr: str,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_forward: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor],
    max_batches: Optional[int] = None,
) -> Dict[Tuple[int, int], float]:
    """Eq. (2)-(4) using arbitrary encoder sequential inside model."""
    m = copy.deepcopy(model).to(device)
    m.train()
    seq = getattr(m, encoder_attr)
    assert isinstance(seq, nn.Sequential)
    conv_ix_list = conv_indices_in_seq(seq)

    grad_sum: Dict[int, Optional[torch.Tensor]] = {fi: None for fi in conv_ix_list}

    n_batches = 0
    for xb, yb in data_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        m.zero_grad(set_to_none=True)
        loss = loss_forward(m, xb, yb)
        loss.backward()
        for fi in conv_ix_list:
            conv = seq[fi]
            assert isinstance(conv, nn.Conv2d)
            g = conv.weight.grad
            if g is None:
                continue
            acc = torch.abs(g.detach())
            prev = grad_sum[fi]
            grad_sum[fi] = acc.clone() if prev is None else prev + acc
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break

    importance: Dict[Tuple[int, int], float] = {}
    for rank, fi in enumerate(conv_ix_list):
        acc = grad_sum[fi]
        seq_cur = getattr(m, encoder_attr)
        conv = seq_cur[fi]
        assert isinstance(conv, nn.Conv2d)
        if acc is None:
            acc = torch.zeros_like(conv.weight)
        d = acc.shape[0]
        w = acc.view(d, -1).mean(dim=1)
        for j in range(d + 1):
            if j == 0:
                p = float(w[0].item())
            elif j == d:
                p = float(w[d - 1].item())
            else:
                p = 0.5 * (float(w[j - 1].item()) + float(w[j].item()))
            importance[(rank, j)] = p
    return importance


def top_n_positions(importance: Dict[Tuple[int, int], float], n_select: int) -> List[InsertionSpec]:
    items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
    specs: List[InsertionSpec] = []
    for (rank, j), _ in items[:n_select]:
        specs.append(InsertionSpec(conv_rank=rank, position=j))
    return specs
