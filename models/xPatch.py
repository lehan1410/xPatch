import torch
import torch.nn as nn
import math

from layers.decomp import DECOMP
from layers.transformer_rmlp import TransformerRMLPNetwork
from layers.revin import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # Parameters
        seq_len = configs.seq_len   # lookback window L
        pred_len = configs.pred_len # prediction length (96, 192, 336, 720)
        c_in = configs.enc_in       # input channels

        # Transformer parameters
        d_model = getattr(configs, 'd_model', 512)  # dimension of model
        nhead = getattr(configs, 'nhead', 8)        # number of heads in multi-head attention
        num_layers = getattr(configs, 'num_layers', 3)  # number of transformer layers
        dropout = getattr(configs, 'dropout', 0.1)      # dropout rate


        # Patching
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)

        # Moving Average
        self.ma_type = configs.ma_type
        alpha = configs.alpha       # smoothing factor for EMA
        beta = configs.beta         # smoothing factor for DEMA

        # Set channel info in configs for TransformerRMLPNetwork
        configs.channel = c_in
        configs.rev = getattr(configs, 'rev', True)  # Enable RevIN by default for RMLP

        self.decomp = DECOMP(self.ma_type, alpha, beta)
        self.net = TransformerRMLPNetwork(seq_len, pred_len, patch_len, stride, padding_patch, d_model, nhead, num_layers, dropout, configs)

    def forward(self, x):
        # x: [Batch, Input, Channel]

        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.ma_type == 'reg':   # If no decomposition
            x = self.net(x, x)
        else:
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)

        # Denormalization
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x