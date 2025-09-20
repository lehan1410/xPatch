import torch
import torch.nn as nn
import torch.nn.functional as F

class EMDDecomp(nn.Module):
    """
    EMD-based decomposition using PyEMD if available.
    season = sum of first n_imfs_keep IMFs; trend = last IMF (approx. residue).
    Falls back to simple moving average if PyEMD not installed.
    """
    def __init__(self, n_imfs_keep: int = 2, fallback_win: int = 15):
        super().__init__()
        self.n_imfs_keep = n_imfs_keep
        self.fallback_win = fallback_win
        try:
            from PyEMD import EMD  # noqa: F401
            self.has_pyemd = True
        except Exception:
            self.has_pyemd = False

    def _fallback_ma(self, x):
        # simple moving average via depthwise conv
        B, T, C = x.shape
        k = self.fallback_win
        pad = (k - 1) // 2
        w = torch.ones(C, 1, k, device=x.device, dtype=x.dtype) / k
        xc = x.transpose(1, 2)  # [B, C, T]
        xc = F.pad(xc, (pad, pad), mode='reflect')
        trend = F.conv1d(xc, w, groups=C).transpose(1, 2)
        season = x - trend
        return season, trend

    def forward(self, x):
        # x: [B, T, C]
        if not self.has_pyemd:
            return self._fallback_ma(x)

        from PyEMD import EMD
        B, T, C = x.shape
        season = torch.zeros_like(x)
        trend = torch.zeros_like(x)
        for b in range(B):
            for c in range(C):
                sig = x[b, :, c].detach().cpu().numpy()
                emd = EMD()
                imfs = emd.emd(sig)  # [n_imfs, T]
                if imfs is None or len(imfs) == 0:
                    s, t = self._fallback_ma(x[b:b+1, :, c:c+1])
                    season[b, :, c] = s[0, :, 0]
                    trend[b, :, c] = t[0, :, 0]
                    continue
                n = imfs.shape[0]
                k = min(self.n_imfs_keep, n)
                s_comp = imfs[:k, :].sum(axis=0)
                t_comp = imfs[-1, :]  # last IMF as trend-like
                season[b, :, c] = torch.from_numpy(s_comp).to(x.device, x.dtype)
                trend[b, :, c] = torch.from_numpy(t_comp).to(x.device, x.dtype)
        return season, trend