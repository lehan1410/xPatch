import torch
from torch import nn
import torch.nn.functional as F

class LearnableWMA(nn.Module):
    def __init__(self, window_sizes, channels):
        super(LearnableWMA, self).__init__()
        self.window_sizes = window_sizes
        self.channels = channels
        self.kernels = nn.ParameterList([
            nn.Parameter(torch.ones(channels, 1, w) / w)
            for w in window_sizes
        ])

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, T]
        outs = []
        for kernel, w in zip(self.kernels, self.window_sizes):
            out = F.conv1d(
                x, kernel, padding=w - 1, groups=C
            )
            out = out[..., :T]
            outs.append(out)
        out = sum(outs) / len(outs)  # Tổng hợp các scale
        out = out.permute(0, 2, 1)  # [B, T, C]
        return out