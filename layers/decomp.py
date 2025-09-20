import torch
from torch import nn

from layers.ema import EMA
from layers.dema import DEMA
from layers.multiband import MultiBandDecomp
from layers.emd import EMDDecomp

class CombinedDecomp(nn.Module):
    """
    Kết hợp MultiBand và EMD với gate động theo dữ liệu (per-channel).
    season = g * season_mb + (1-g) * season_emd
    trend  = g * trend_mb  + (1-g) * trend_emd
    g = sigmoid(bias + scale * mean_t(x)), mean_t là trung bình theo thời gian.
    """
    def __init__(self, channels: int, mb_k_small: int = 7, mb_k_large: int = 31, emd_imfs: int = 2,
                 gate_bias_init: float = 1.0):
        super().__init__()
        self.mb = MultiBandDecomp(channels, k_small=mb_k_small, k_large=mb_k_large)
        self.emd = EMDDecomp(n_imfs_keep=emd_imfs)

        # Tham số gate per-channel: g = sigmoid(bias + scale * pooled_x)
        self.gate_bias = nn.Parameter(torch.full((1, 1, channels), gate_bias_init))  # nghiêng về Multi-band
        self.gate_scale = nn.Parameter(torch.zeros(1, 1, channels))                  # khởi tạo 0 -> g ~ sigmoid(bias)

    def forward(self, x):
        # x: [B, T, C]
        s_mb, t_mb, _ = self.mb(x)   # khả vi
        s_emd, t_emd = self.emd(x)   # không khả vi theo x, nhưng hợp lệ cho forward

        # Gate động theo dữ liệu: trung bình theo thời gian
        pooled = x.mean(dim=1, keepdim=True)                  # [B, 1, C]
        g = torch.sigmoid(self.gate_bias + self.gate_scale * pooled)  # [B, 1, C], broadcast theo T

        season = g * s_mb + (1.0 - g) * s_emd
        trend  = g * t_mb + (1.0 - g) * t_emd
        return season, trend

class DECOMP(nn.Module):
    """
    Hỗ trợ: 'reg', 'ema', 'dema', 'multiband', 'emd', 'multi_emd'
    """
    def __init__(self, ma_type, alpha=None, beta=None, seq_len=None, enc_in=None,
                 mb_k_small: int = 7, mb_k_large: int = 31, emd_imfs: int = 2):
        super(DECOMP, self).__init__()
        self.ma_type = ma_type
        if ma_type == 'ema':
            self.ma = EMA(alpha)
        elif ma_type == 'dema':
            self.ma = DEMA(alpha, beta)
        elif ma_type == 'multiband':
            assert enc_in is not None, "enc_in (channels) required for multiband"
            self.ma = MultiBandDecomp(enc_in, k_small=mb_k_small, k_large=mb_k_large)
        elif ma_type == 'emd':
            self.ma = EMDDecomp(n_imfs_keep=emd_imfs)
        elif ma_type == 'multi_emd':
            assert enc_in is not None, "enc_in (channels) required for multi_emd"
            self.ma = CombinedDecomp(enc_in, mb_k_small=mb_k_small, mb_k_large=mb_k_large, emd_imfs=emd_imfs)
        elif ma_type == 'reg':
            self.ma = None
        else:
            raise ValueError(f"Unknown ma_type: {ma_type}")

    def forward(self, x):
        # x: [B, T, C]
        if self.ma_type == 'reg':
            return x, x
        if self.ma_type in ('ema', 'dema'):
            moving_average = self.ma(x)
            res = x - moving_average
            return res, moving_average
        return self.ma(x)