"""Side information hiding (SIH): embed interference layout into a side filter (LSB, keyed)."""

from __future__ import annotations

import copy
import hashlib
import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from core.gfi import _features_indices_of_convs, insert_one_filter
from core.gfi_seq import conv_indices_in_seq, insert_one_filter_seq
from models.vgg11 import VGG11


def _key_stream(key: bytes, nbytes: int) -> bytes:
    out = bytearray()
    ctr = 0
    while len(out) < nbytes:
        h = hashlib.sha256(key + struct.pack("<I", ctr)).digest()
        out.extend(h)
        ctr += 1
    return bytes(out[:nbytes])


def pack_conv_masks(conv_masks: Dict[int, torch.Tensor]) -> bytes:
    """Concatenate 0/1 bits per conv rank in sorted rank order."""
    bits: List[int] = []
    for rank in sorted(conv_masks.keys()):
        row = conv_masks[rank].to(torch.uint8).tolist()
        bits.extend(int(b) for b in row)
    out = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i : i + 8]
        while len(chunk) < 8:
            chunk.append(0)
        v = 0
        for j, b in enumerate(chunk):
            v |= (b & 1) << j
        out.append(v)
    return bytes(out)


def unpack_conv_masks(data: bytes, shapes: Dict[int, int]) -> Dict[int, torch.Tensor]:
    """Recreate conv_masks given per-rank channel counts (original stego layout before side removal)."""
    bits: List[int] = []
    for byte in data:
        for j in range(8):
            bits.append((byte >> j) & 1)
    masks: Dict[int, torch.Tensor] = {}
    idx = 0
    for rank in sorted(shapes.keys()):
        n = shapes[rank]
        masks[rank] = torch.tensor(bits[idx : idx + n], dtype=torch.uint8)
        idx += n
    return masks


def encrypt_payload(payload: bytes, key: bytes) -> bytes:
    ks = _key_stream(key, len(payload))
    return bytes(a ^ b for a, b in zip(payload, ks))


def embed_bits_lsb_(tensor: torch.Tensor, bits: List[int], start: int = 0) -> int:
    """Write LSBs into flattened tensor from float weights (Guan et al. style). Returns next bit index."""
    flat = tensor.view(-1)
    scale = 1_000_000.0
    t = 0
    for i in range(flat.numel()):
        if start + t >= len(bits):
            break
        val = flat[i].item()
        ival = int(round(val * scale))
        bit = bits[start + t] & 1
        ival = (ival & ~1) | bit
        flat[i] = float(ival) / scale
        t += 1
    return start + t


@dataclass
class SideInfo:
    conv_rank: int
    channel_index: int  # output channel index of side filter in that conv layer
    payload_bytes: int  # length of encrypted B' in bytes


def side_insertion_from_key(key: bytes, num_conv_layers: int, max_positions_per_layer: List[int]) -> Tuple[int, int]:
    """Pick (conv_rank, position) deterministically."""
    h = hashlib.sha256(key + b"sidepos").digest()
    rank = int.from_bytes(h[:2], "little") % num_conv_layers
    mp = max(1, max_positions_per_layer[rank])
    pos = int.from_bytes(h[2:4], "little") % mp
    return rank, pos


def insert_side_filter_and_embed(
    model: VGG11,
    conv_masks: Dict[int, torch.Tensor],
    key: bytes,
    rng: torch.Generator,
) -> Tuple[VGG11, SideInfo]:
    conv_indices = _features_indices_of_convs(model.features)
    max_pos = [model.features[i].out_channels + 1 for i in conv_indices]  # type: ignore[union-attr]
    rank, pos = side_insertion_from_key(key, len(conv_indices), max_pos)
    m = copy.deepcopy(model)
    m, _ = insert_one_filter(m, rank, pos, rng)
    payload = pack_conv_masks(conv_masks)
    enc = encrypt_payload(payload, key)
    bits: List[int] = []
    for b in enc:
        for j in range(8):
            bits.append((b >> j) & 1)
    fi = conv_indices[rank]
    conv = m.features[fi]
    assert isinstance(conv, nn.Conv2d)
    used = embed_bits_lsb_(conv.weight.data, bits, 0)
    if conv.bias is not None and used < len(bits):
        embed_bits_lsb_(conv.bias.data, bits, used)
    return m, SideInfo(conv_rank=rank, channel_index=pos, payload_bytes=len(enc))


def extract_payload_from_side(
    model: VGG11,
    side: SideInfo,
    key: bytes,
    shapes: Dict[int, int],
) -> Dict[int, torch.Tensor]:
    fi = _features_indices_of_convs(model.features)[side.conv_rank]
    conv = model.features[fi]
    assert isinstance(conv, nn.Conv2d)
    scale = 1_000_000.0
    bits: List[int] = []
    for v in conv.weight.data.view(-1):
        ival = int(round(v.item() * scale))
        bits.append(ival & 1)
        if len(bits) >= 8 * side.payload_bytes:
            break
    if conv.bias is not None and len(bits) < 8 * side.payload_bytes:
        for v in conv.bias.data.view(-1):
            ival = int(round(v.item() * scale))
            bits.append(ival & 1)
            if len(bits) >= 8 * side.payload_bytes:
                break
    bits = bits[: 8 * side.payload_bytes]
    out_bytes = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i : i + 8]
        while len(chunk) < 8:
            chunk.append(0)
        v = 0
        for j, b in enumerate(chunk):
            v |= (b & 1) << j
        out_bytes.append(v)
    dec = encrypt_payload(bytes(out_bytes), key)
    return unpack_conv_masks(dec, shapes)


def insert_side_filter_and_embed_encoder(
    model: nn.Module,
    encoder_attr: str,
    conv_masks: Dict[int, torch.Tensor],
    key: bytes,
    rng: torch.Generator,
) -> Tuple[nn.Module, SideInfo]:
    """SIH for models that keep convolutions in an `nn.Sequential` (e.g. DualHeadDnCNN.encoder)."""
    m = copy.deepcopy(model)
    enc = getattr(m, encoder_attr)
    assert isinstance(enc, nn.Sequential)
    conv_ix = conv_indices_in_seq(enc)
    max_pos = [enc[i].out_channels + 1 for i in conv_ix]  # type: ignore[index]
    rank, pos = side_insertion_from_key(key, len(conv_ix), max_pos)
    new_enc = insert_one_filter_seq(enc, rank, pos, rng)
    setattr(m, encoder_attr, new_enc)
    payload = pack_conv_masks(conv_masks)
    enc_seq = getattr(m, encoder_attr)
    conv_ix = conv_indices_in_seq(enc_seq)
    fi = conv_ix[rank]
    conv = enc_seq[fi]
    assert isinstance(conv, nn.Conv2d)
    enc_payload = encrypt_payload(payload, key)
    bits: List[int] = []
    for b in enc_payload:
        for j in range(8):
            bits.append((b >> j) & 1)
    used = embed_bits_lsb_(conv.weight.data, bits, 0)
    if conv.bias is not None and used < len(bits):
        embed_bits_lsb_(conv.bias.data, bits, used)
    return m, SideInfo(conv_rank=rank, channel_index=pos, payload_bytes=len(enc_payload))


def extract_payload_from_side_encoder(
    model: nn.Module,
    side: SideInfo,
    key: bytes,
    shapes: Dict[int, int],
    encoder_attr: str = "encoder",
) -> Dict[int, torch.Tensor]:
    enc = getattr(model, encoder_attr)
    fi = conv_indices_in_seq(enc)[side.conv_rank]
    conv = enc[fi]
    assert isinstance(conv, nn.Conv2d)
    scale = 1_000_000.0
    bits: List[int] = []
    for v in conv.weight.data.view(-1):
        ival = int(round(v.item() * scale))
        bits.append(ival & 1)
        if len(bits) >= 8 * side.payload_bytes:
            break
    if conv.bias is not None and len(bits) < 8 * side.payload_bytes:
        for v in conv.bias.data.view(-1):
            ival = int(round(v.item() * scale))
            bits.append(ival & 1)
            if len(bits) >= 8 * side.payload_bytes:
                break
    bits = bits[: 8 * side.payload_bytes]
    out_bytes = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i : i + 8]
        while len(chunk) < 8:
            chunk.append(0)
        v = 0
        for j, b in enumerate(chunk):
            v |= (b & 1) << j
        out_bytes.append(v)
    dec = encrypt_payload(bytes(out_bytes), key)
    return unpack_conv_masks(dec, shapes)
