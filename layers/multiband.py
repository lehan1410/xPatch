import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSmoother(nn.Module):
    """Learnable depthwise low-pass smoother with normalized kernel per channel."""
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.zeros(channels, 1, kernel_size))
        nn.init.constant_(self.weight, 0.0)

    def forward(self, x):
        # x: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        w = torch.softmax(self.weight, dim=-1)       # positive, normalized kernel
        pad = (self.kernel_size - 1) // 2
        x = F.pad(x, (pad, pad), mode='reflect')
        y = F.conv1d(x, w, bias=None, stride=1, padding=0, groups=self.channels)
        return y.transpose(1, 2)  # [B, T, C]

class MultiBandDecomp(nn.Module):
    """
    Multi-band: high = x - LP(k_s), mid = LP(k_s) - LP(k_l), low = LP(k_l)
    season = high + mid, trend = low
    """
    def __init__(self, channels: int, k_small: int = 7, k_large: int = 31):
        super().__init__()
        assert k_small % 2 == 1 and k_large % 2 == 1, "kernel sizes should be odd"
        assert k_small < k_large, "k_small must be < k_large"
        self.lp_small = DepthwiseSmoother(channels, k_small)
        self.lp_large = DepthwiseSmoother(channels, k_large)

    def forward(self, x):
        # x: [B, T, C]
        lp_s = self.lp_small(x)
        lp_l = self.lp_large(x)
        high = x - lp_s
        mid = lp_s - lp_l
        low = lp_l
        season = high + mid
        trend = low
        return season, trend, (high, mid, low)