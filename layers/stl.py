import torch
import numpy as np
from statsmodels.tsa.seasonal import STL

def stl_decompose(x, period):
    """
    STL decomposition for time series.
    x: [Batch, Input, Channel] (torch.Tensor)
    period: seasonality period (int)
    Returns: seasonal, trend, resid (same shape as x)
    """
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x

    B, T, C = x_np.shape
    seasonal = np.zeros_like(x_np)
    trend = np.zeros_like(x_np)
    resid = np.zeros_like(x_np)
    for b in range(B):
        for c in range(C):
            stl = STL(x_np[b, :, c], period=period, robust=True)
            res = stl.fit()
            seasonal[b, :, c] = res.seasonal
            trend[b, :, c] = res.trend
            resid[b, :, c] = res.resid
    return (
        torch.tensor(seasonal, dtype=x.dtype, device=x.device),
        torch.tensor(trend, dtype=x.dtype, device=x.device),
        torch.tensor(resid, dtype=x.dtype, device=x.device)
    )