import math
import torch
import torch.nn.functional as F
from torch import nn

from layers.ema import EMA
from layers.dema import DEMA

__all__ = ["MultiScaleTCNDecompose", "DECOMP"]

class TCNScale(nn.Module):
    """
    Small TCN stack that achieves a target receptive field (approx kernel size).
    Input: x of shape [N, L] -> processes as [N, 1, L] and returns [N, L].
    Shared weights across channels (applied on flattened B*C batch).
    """
    def __init__(self, target_kernel, kernel_size=3, hidden_channels=1, activation=nn.SiLU):
        super(TCNScale, self).__init__()
        assert target_kernel >= 1 and kernel_size >= 3 and kernel_size % 2 == 1
        self.kernel_size = kernel_size
        self.activation = activation()
        # compute number of dilated conv layers needed to reach receptive field >= target_kernel
        rf = 1
        dilation = 1
        layers = []
        while rf < target_kernel:
            layers.append(dilation)
            rf += (kernel_size - 1) * dilation
            dilation *= 2
        self.dilations = layers
        convs = []
        for d in self.dilations:
            conv = nn.Conv1d(in_channels=1, out_channels=1,
                             kernel_size=kernel_size, padding=d, dilation=d, bias=True)
            convs.append(conv)
        self.convs = nn.ModuleList(convs)
        # small residual scale parameter to stabilise training
        self.res_scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_flat):
        # x_flat: [N, L]
        x = x_flat.unsqueeze(1)  # [N,1,L]
        out = x
        for conv in self.convs:
            y = conv(out)
            y = self.activation(y)
            # residual in channel dimension
            out = out + self.res_scale * y
        return out.squeeze(1)  # [N, L]


class MultiScaleTCNDecompose(nn.Module):
    """
    Multi-scale decomposition with three TCN scales (long, med, short).
    Input x: [B, L, C] -> outputs seasonal, trend, cyclic, irregular each [B, L, C].
    The TCNs operate on flattened batch B*C so weights are shared across channels.
    """
    def __init__(self, kernel_sizes=(101, 31, 7)):
        super(MultiScaleTCNDecompose, self).__init__()
        assert len(kernel_sizes) == 3, "Expect three kernel sizes (long, med, short)"
        for k in kernel_sizes:
            assert k % 1 == 0 and k > 0
        self.k_long, self.k_med, self.k_short = kernel_sizes

        # build TCN stacks per scale
        self.scale_long = TCNScale(self.k_long)
        self.scale_med = TCNScale(self.k_med)
        self.scale_short = TCNScale(self.k_short)

    def forward(self, x):
        # expect x: [B, L, C]
        if x.dim() != 3:
            raise ValueError("MultiScaleTCNDecompose expects input shape [B, L, C]")

        B, L, C = x.shape
        # reshape to [B*C, L] for per-channel independent processing
        x_perm = x.permute(0, 2, 1).reshape(B * C, L)  # [N, L]

        s_long = self.scale_long(x_perm)   # [N, L]
        s_med = self.scale_med(x_perm)     # [N, L]
        s_short = self.scale_short(x_perm) # [N, L]

        # components in flattened domain
        trend_flat = s_long
        cyclic_flat = s_med - s_long
        seasonal_flat = s_short - s_med
        irregular_flat = x_perm - s_short

        # reshape back to [B, L, C]
        def restore(z):
            return z.reshape(B, C, L).permute(0, 2, 1)  # [B, L, C]

        seasonal = restore(seasonal_flat)
        trend = restore(trend_flat)
        cyclic = restore(cyclic_flat)
        irregular = restore(irregular_flat)

        return seasonal, trend, cyclic, irregular


class DECOMP(nn.Module):
    """
    Series decomposition block that wraps EMA/DEMA or learnable MultiScaleTCNDecompose.
    """
    def __init__(self, ma_type, alpha, beta, learn_kernels=(101, 31, 7)):
        super(DECOMP, self).__init__()
        self.ma_type = ma_type
        if ma_type == 'ema':
            self.ma = EMA(alpha)
        elif ma_type == 'dema':
            self.ma = DEMA(alpha, beta)
        elif ma_type == 'learn':
            self.ma = MultiScaleTCNDecompose(kernel_sizes=learn_kernels)
        else:
            raise ValueError(f"Unknown ma_type: {ma_type}")

    def forward(self, x):
        if self.ma_type in ('ema', 'dema'):
            moving_average = self.ma(x)
            res = x - moving_average
            return res, moving_average
        else:  # 'learn'
            seasonal, trend, cyclic, irregular = self.ma(x)
            return seasonal, trend, cyclic, irregular