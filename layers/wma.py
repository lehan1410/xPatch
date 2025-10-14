import torch
from torch import nn
import torch.nn.functional as F

class WMA(nn.Module):
    """
    Weighted Moving Average (WMA) block to highlight the trend of time series
    (Fast version using conv1d)
    """
    def __init__(self, window_size):
        super(WMA, self).__init__()
        self.window_size = window_size
        weights = torch.arange(1, window_size + 1).float()
        weights = weights / weights.sum()
        self.register_buffer('weights', weights.view(1, 1, -1))  # [1, 1, window_size]

    def forward(self, x):
        # x: [Batch, Input, Channel]
        B, T, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, T]
        out = F.conv1d(
            x, self.weights.expand(C, -1, -1), 
            padding=self.window_size - 1, 
            groups=C
        )
        out = out[..., :T]  # Cắt về đúng chiều dài đầu vào
        out = out.permute(0, 2, 1)  # [B, T, C]
        return out