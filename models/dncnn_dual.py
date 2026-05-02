"""DnCNN-style encoder with frozen secret denoise head + trainable segmentation head (inter-task stego)."""

from __future__ import annotations

from typing import List

import torch.nn as nn


def build_encoder(num_conv_layers: int, width: int, in_ch: int = 3) -> nn.Sequential:
    """Stack of Conv-BN-ReLU blocks; each Conv2d uses `width` output channels after the first."""
    layers: List[nn.Module] = []
    for i in range(num_conv_layers):
        cin = in_ch if i == 0 else width
        layers.append(nn.Conv2d(cin, width, kernel_size=3, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(width))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class DualHeadDnCNN(nn.Module):
    """
    Encoder produces width-channel maps; secret_head maps to RGB denoising output.
    Stego uses stego_head for dense segmentation logits (same spatial size).

    Keep insertions out of the **last encoder conv rank** so encoder output stays `width`-dim
    (matches frozen secret_head Conv2d(width -> RGB)).
    """

    def __init__(
        self,
        num_seg_classes: int,
        num_encoder_convs: int = 8,
        width: int = 64,
        in_channels: int = 3,
        init_weights: bool = True,
    ):
        super().__init__()
        self.width = width
        self.encoder = build_encoder(num_encoder_convs, width, in_ch=in_channels)
        self.secret_head = nn.Conv2d(width, in_channels, kernel_size=3, padding=1, bias=False)
        self.stego_head = nn.Conv2d(width, num_seg_classes, kernel_size=3, padding=1, bias=True)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_stego(self, x):
        z = self.encoder(x)
        return self.stego_head(z)

    def forward_secret(self, x):
        z = self.encoder(x)
        return self.secret_head(z)
