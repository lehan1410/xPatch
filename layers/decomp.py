import torch
from torch import nn

from layers.ema import EMA
from layers.dema import DEMA
import torch.nn.functional as F

class MultiScaleDecompose(nn.Module):
    """
    Multi-scale learnable decomposition.
    Given input x shape [B, L, C] (same as project convention [Batch, Input, Channel]),
    returns seasonal, trend, cyclic, irregular each with same shape.
    Uses three odd kernel sizes (long, med, short) to produce multiscale smoothers.
    """
    def __init__(self, kernel_sizes=(101, 31, 7)):
        super(MultiScaleDecompose, self).__init__()
        assert len(kernel_sizes) == 3, "Expect three kernel sizes (long, med, short)"
        for k in kernel_sizes:
            assert k % 2 == 1, "kernel sizes must be odd"
        self.k_long, self.k_med, self.k_short = kernel_sizes

        # learnable logits for each kernel -> softmax to get normalized smoothing kernel
        self.logits_long = nn.Parameter(torch.randn(self.k_long))
        self.logits_med = nn.Parameter(torch.randn(self.k_med))
        self.logits_short = nn.Parameter(torch.randn(self.k_short))

    def _smooth(self, x_flat, logits, k):
        # x_flat: [N, L]
        w = F.softmax(logits, dim=0).view(1, 1, k).to(x_flat.dtype).to(x_flat.device)
        x_in = x_flat.unsqueeze(1)  # [N,1,L]
        pad = k // 2
        out = F.conv1d(x_in, w, padding=pad)  # [N,1,L]
        return out.squeeze(1)  # [N, L]

    def forward(self, x):
        # expect x: [B, Input, Channel]
        if x.dim() != 3:
            raise ValueError("MultiScaleDecompose expects input shape [B, Input, Channel]")

        B, L, C = x.shape
        # reshape to [B*C, L] for per-channel independent smoothing
        x_perm = x.permute(0,2,1).reshape(B * C, L)  # [N, L]

        s_long = self._smooth(x_perm, self.logits_long, self.k_long)
        s_med = self._smooth(x_perm, self.logits_med, self.k_med)
        s_short = self._smooth(x_perm, self.logits_short, self.k_short)

        # components in flattened domain
        trend_flat = s_long
        cyclic_flat = s_med - s_long
        seasonal_flat = s_short - s_med
        irregular_flat = x_perm - s_short

        # reshape back to [B, L, C]
        def restore(z):
            return z.reshape(B, C, L).permute(0,2,1)  # [B, L, C]

        seasonal = restore(seasonal_flat)
        trend = restore(trend_flat)
        cyclic = restore(cyclic_flat)
        irregular = restore(irregular_flat)

        return seasonal, trend, cyclic, irregular

class DECOMP(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, ma_type, alpha, beta, learn_kernels=(101,31,7)):
        super(DECOMP, self).__init__()
        self.ma_type = ma_type
        if ma_type == 'ema':
            self.ma = EMA(alpha)
        elif ma_type == 'dema':
            self.ma = DEMA(alpha, beta)
        elif ma_type == 'learn':
            self.ma = MultiScaleDecompose(kernel_sizes=learn_kernels)

    def forward(self, x):
        if self.ma_type in ('ema', 'dema'):
            moving_average = self.ma(x)
            res = x - moving_average
            return res, moving_average
        else:  # 'learn'
            seasonal, trend, cyclic, irregular = self.ma(x)
            return seasonal, trend, cyclic, irregular