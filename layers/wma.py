import torch
from torch import nn

class WMA(nn.Module):
    """
    Weighted Moving Average (WMA) block to highlight the trend of time series
    """
    def __init__(self, window_size):
        super(WMA, self).__init__()
        self.window_size = window_size
        # Trọng số tăng dần cho các điểm gần hiện tại hơn
        self.register_buffer('weights', torch.arange(1, window_size + 1).float())

    def forward(self, x):
        # x: [Batch, Input, Channel]
        B, T, C = x.shape
        out = torch.zeros_like(x)
        for i in range(T):
            start = max(0, i - self.window_size + 1)
            window = x[:, start:i+1, :]  # [B, w, C]
            w = self.weights[-window.shape[1]:].view(1, -1, 1)
            weighted = window * w
            out[:, i, :] = weighted.sum(dim=1) / w.sum()
        return out