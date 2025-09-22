import torch
from torch import nn
import torch.nn.functional as F


class MultiScaleDWConv(nn.Module):
    """Lightweight multiscale depthwise conv with efficient fusion."""
    def __init__(self, channels: int, dilations=(1, 2, 4), kernel_size: int = 3, 
                 norm_type: str = 'bn', dropout: float = 0.05):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.scales = tuple(dilations)  # Reduced from 5 to 3 scales
        self.S = len(self.scales)
        
        # Single grouped conv for all scales (more efficient)
        pad = kernel_size // 2
        self.conv = nn.Conv1d(channels, channels * self.S, kernel_size=kernel_size, 
                             padding=pad, groups=channels)
        
        if norm_type == 'bn':
            self.bn = nn.BatchNorm1d(channels * self.S)
        else:
            self.bn = nn.Identity()
        
        self.act = nn.GELU()
        
        # Simple linear fusion (no learnable weights)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x_bci):
        # x_bci: [B, C, T]
        B, C, T = x_bci.shape
        
        # Single conv operation
        y = self.conv(x_bci)  # [B, C*S, T]
        y = self.act(self.bn(y))
        
        # Reshape and simple average
        y = y.view(B, C, self.S, T)
        fused = y.mean(dim=2)  # Simple average across scales
        fused = self.drop(fused)
        
        # Residual connection
        return x_bci + fused


class DirectLinearStream(nn.Module):
    """Simplified direct linear mapping for speed."""
    def __init__(self, seq_len: int, pred_len: int, dropout: float = 0.05):
        super().__init__()
        # Simple direct mapping without intermediate layers
        self.linear = nn.Linear(seq_len, pred_len)
        self.dropout = nn.Dropout(dropout)
        
        # Efficient initialization
        nn.init.xavier_uniform_(self.linear.weight, gain=0.8)
    
    def forward(self, x_bci: torch.Tensor) -> torch.Tensor:
        # x_bci: [B, C, T] -> direct mapping
        x = self.dropout(x_bci)
        x = self.linear(x)  # [B, C, pred_len]
        return x


class InterChannelLinear(nn.Module):
    """Lightweight inter-channel mixer with minimal overhead."""
    def __init__(self, seq_len: int, pred_len: int, c_in: int, rank: int = None, dropout: float = 0.05):
        super().__init__()
        self.time_proj = nn.Linear(seq_len, pred_len)
        r = min(8, c_in) if rank is None else min(rank, c_in)  # Keep rank small for speed
        
        # Simple low-rank channel mixing
        self.W1 = nn.Linear(c_in, r, bias=False)
        self.W2 = nn.Linear(r, c_in, bias=False)
        self.drop = nn.Dropout(dropout)
        
        # Simple initialization
        nn.init.xavier_uniform_(self.time_proj.weight)
        nn.init.orthogonal_(self.W1.weight)
        nn.init.orthogonal_(self.W2.weight)
        
    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        # x_bct: [B, C, T]
        y = self.time_proj(x_bct)  # [B, C, pred_len]
        # Channel mixing per time step
        y_perm = y.transpose(1, 2)  # [B, pred_len, C]
        y_mix = self.W2(self.drop(self.W1(y_perm)))
        
        # Simple residual
        y_out = (y_perm + 0.1 * y_mix).transpose(1, 2)
        return y_out


class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, c_in: int):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.c_in = c_in

        # Multiscale preprocessing (lightweight with fewer scales)
        self.ms_season = MultiScaleDWConv(c_in, dilations=(1, 2, 4), kernel_size=3, dropout=0.05)
        self.ms_trend = MultiScaleDWConv(c_in, dilations=(1, 3, 6), kernel_size=3, dropout=0.05)

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

        # Inter-channel stream (lightweight)
        self.ic_linear = InterChannelLinear(seq_len=seq_len, pred_len=pred_len, c_in=c_in, rank=8, dropout=0.05)
        
        # Direct linear stream (simplified)
        self.direct_linear = DirectLinearStream(seq_len=seq_len, pred_len=pred_len, dropout=0.05)

        # Simplified fusion (no learnable weights, just concatenation)
        self.fc8 = nn.Linear(pred_len * 4, pred_len)
        
        # Simple initialization
        nn.init.xavier_uniform_(self.fc8.weight, gain=0.5)

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

        # Simple streams concatenation (no learnable weights)
        x_ic_flat = torch.reshape(x_ic, (B * C, self.pred_len))
        x_direct_flat = torch.reshape(x_direct, (B * C, self.pred_len))
        
        x = torch.cat((s, t, x_ic_flat, x_direct_flat), dim=1)
        x = self.fc8(x)

        # Channel concatination
        x = torch.reshape(x, (B, C, self.pred_len)) # [Batch, Channel, Output]

        x = x.permute(0,2,1) # to [Batch, Output, Channel]

        return x