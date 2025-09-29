import torch
from torch import nn

from layers.ema import EMA
from layers.dema import DEMA
from layers.wma import WMA
from layers.stl import stl_decompose  # Thêm dòng này

class DECOMP(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, ma_type, alpha, beta, period=24):
        super(DECOMP, self).__init__()
        self.ma_type = ma_type
        self.period = period
        if ma_type == 'ema':
            self.ma = EMA(alpha)
        elif ma_type == 'dema':
            self.ma = DEMA(alpha, beta)
        elif ma_type == 'wma':
            self.ma = WMA(window_size=period)
        elif ma_type == 'stl':
            self.ma = None  # Không cần module cho STL

    def forward(self, x):
        if self.ma_type == 'stl':
            seasonal, trend, resid = stl_decompose(x, period=self.period)
            return seasonal, trend, resid  # trend là moving_average, resid là phần còn lại
        else:
            moving_average = self.ma(x)
            res = x - moving_average
            return res, moving_average