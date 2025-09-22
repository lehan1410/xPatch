import torch
from torch import nn
import torch.nn.functional as F


class MultiScaleDWConv(nn.Module):
    """Multiscale depthwise conv with linear inner/inter scale fusion (no attention)."""
    def __init__(self, channels: int, dilations=(1, 2, 4, 8), kernel_size: int = 3, 
                 norm_type: str = 'bn', dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.scales = tuple(dilations)
        self.S = len(self.scales)
        
        # Per-scale depthwise convs
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for d in self.scales:
            pad = d * (kernel_size // 2)
            self.convs.append(
                nn.Conv1d(channels, channels, kernel_size=kernel_size, 
                         padding=pad, dilation=d, groups=channels)
            )
            if norm_type == 'bn':
                self.bns.append(nn.BatchNorm1d(channels))
            elif norm_type == 'ln':
                self.bns.append(nn.GroupNorm(1, channels))  # LayerNorm-like
            else:
                self.bns.append(nn.Identity())
        
        self.act = nn.GELU()
        
        # Linear inter-scale fusion (no attention)
        self.scale_weights = nn.Parameter(torch.ones(self.S) / self.S)  # learnable weights
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x_bci):
        # x_bci: [B, C, T]
        ys = []
        for conv, bn in zip(self.convs, self.bns):
            y = conv(x_bci)
            y = self.act(bn(y))
            ys.append(y)
        
        # Linear combination with learnable weights
        weights = F.softmax(self.scale_weights, dim=0)  # normalize to sum=1
        fused = sum(w * y for w, y in zip(weights, ys))
        fused = self.drop(fused)
        
        # Residual connection
        return x_bci + fused


class DirectLinearStream(nn.Module):
    """Direct linear mapping inspired by RLinear - simple but effective."""
    def __init__(self, seq_len: int, pred_len: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(seq_len, seq_len)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(seq_len, pred_len)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_bci: torch.Tensor) -> torch.Tensor:
        # x_bci: [B, C, T] -> permute to [B, C, T] for per-channel processing
        x = self.linear1(x_bci)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)  # [B, C, pred_len]
        return x


class InterChannelLinear(nn.Module):
    """Pure linear inter-channel mixer with low-rank factorization."""
    def __init__(self, seq_len: int, pred_len: int, c_in: int, rank: int = None, dropout: float = 0.1):
        super().__init__()
        self.time_proj = nn.Linear(seq_len, pred_len)
        r = min(8, c_in) if rank is None else min(rank, c_in)
        # Low-rank channel mixing
        self.W1 = nn.Linear(c_in, r, bias=False)
        self.W2 = nn.Linear(r, c_in, bias=False)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        # x_bct: [B, C, T]
        y = self.time_proj(x_bct)  # [B, C, pred_len]
        # Channel mixing per time step
        y_perm = y.transpose(1, 2)  # [B, pred_len, C]
        y_mix = self.W2(self.drop(self.W1(y_perm)))
        y_out = (y_perm + y_mix).transpose(1, 2)  # residual + back to [B, C, pred_len]
        return y_out


class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, c_in: int):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.c_in = c_in

        # Multiscale preprocessing (applied before channel split)
        self.ms_season = MultiScaleDWConv(c_in, dilations=(1, 2, 4, 8, 16), kernel_size=3, dropout=0.1)
        self.ms_trend = MultiScaleDWConv(c_in, dilations=(1, 3, 6, 12, 24), kernel_size=5, dropout=0.1)

        # Non-linear Stream
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len)//stride + 1
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            self.patch_num += 1

        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)
        
        # CNN Depthwise
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream
        self.fc2 = nn.Linear(self.dim, patch_len)

        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # Linear Stream
        # MLP
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # Inter-channel stream (linear mixing)
        self.ic_linear = InterChannelLinear(seq_len=seq_len, pred_len=pred_len, c_in=c_in, rank=8, dropout=0.1)
        
        # Direct linear stream (RLinear-inspired)
        self.direct_linear = DirectLinearStream(seq_len=seq_len, pred_len=pred_len, dropout=0.1)

        # Streams Concatenation (4 streams: s, t, inter-channel, direct)
        self.fc8 = nn.Linear(pred_len * 4, pred_len)

    def forward(self, s, t):
        # x: [Batch, Input, Channel]
        # s - seasonality
        # t - trend
        
        s = s.permute(0,2,1) # to [Batch, Channel, Input]
        t = t.permute(0,2,1) # to [Batch, Channel, Input]

        # Multiscale temporal processing (before channel split)
        s = self.ms_season(s)  # [B, C, I]
        t = self.ms_trend(t)   # [B, C, I]

        # Inter-channel stream (uses combined features)
        u = 0.5 * (s + t)
        x_ic = self.ic_linear(u)  # [B, C, pred_len]
        
        # Direct linear stream (RLinear-style on combined input)
        x_direct = self.direct_linear(u)  # [B, C, pred_len]
        
        # Channel split for channel independence
        B = s.shape[0] # Batch size
        C = s.shape[1] # Channel size
        I = s.shape[2] # Input size
        s = torch.reshape(s, (B*C, I)) # [Batch and Channel, Input]
        t = torch.reshape(t, (B*C, I)) # [Batch and Channel, Input]

        # Non-linear Stream
        # Patching
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: [Batch and Channel, Patch_num, Patch_len]
        
        # Patch Embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        # CNN Depthwise
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Residual Stream
        res = self.fc2(res)
        s = s + res

        # CNN Pointwise
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # Flatten Head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        # Linear Stream
        # MLP
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)

        # Streams Concatenation (4 streams)
        x_ic_flat = torch.reshape(x_ic, (B * C, self.pred_len))
        x_direct_flat = torch.reshape(x_direct, (B * C, self.pred_len))
        x = torch.cat((s, t, x_ic_flat, x_direct_flat), dim=1)
        x = self.fc8(x)

        # Channel concatination
        x = torch.reshape(x, (B, C, self.pred_len)) # [Batch, Channel, Output]

        x = x.permute(0,2,1) # to [Batch, Output, Channel]

        return x