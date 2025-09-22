import torch
from torch import nn
import torch.nn.functional as F
from .conv_backbone import ConvBackbone


class GEGLULinear(nn.Module):
    """Linear layer with GEGLU gating: proj to 2*out, split, GELU on gate, elementwise product."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x):
        u, v = self.proj(x).chunk(2, dim=-1)
        return u * F.gelu(v)


class ChannelSEGate(nn.Module):
    """Inter-channel gating without value mixing: uses cross-channel MLP to produce per-channel scales in [0,1]."""
    def __init__(self, channels: int, hidden_ratio: int = 4):
        super().__init__()
        hidden = max(4, channels // hidden_ratio)
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, channels)
        # gentle gate init: bias ~ 1.5 -> sigmoid ≈ 0.82
        nn.init.constant_(self.fc2.bias, 1.5)
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.7)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.7)

    def forward(self, x_bci):
        # x_bci: [B, C, T]
        m = x_bci.mean(dim=-1)  # [B, C]
        g = torch.sigmoid(self.fc2(self.act(self.fc1(m))))  # [B, C]
        return g.unsqueeze(-1)  # [B, C, 1]


def _make_norm(norm_type: str, channels: int):
    norm_type = (norm_type or 'bn').lower()
    if norm_type == 'bn':
        return nn.BatchNorm1d(channels)
    if norm_type == 'in':
        return nn.InstanceNorm1d(channels, affine=True)
    if norm_type == 'gn':
        # fallback to LayerNorm-like behavior across channels/time by using 1 group
        return nn.GroupNorm(1, channels)
    if norm_type == 'none':
        return nn.Identity()
    return nn.BatchNorm1d(channels)


class MultiScaleDWConv(nn.Module):
    """Depthwise temporal conv pyramid with dilations + robust inter-scale fusion.

    - Inner-scale: per-scale depthwise Conv1d (+BN+GELU)
    - Inter-scale: temperature-scaled softmax attention over scales using mean+std pooling across time
    - Residual fusion: out = x + alpha * (avg_scales + beta * attn_mix)
    """
    def __init__(self, channels: int, dilations=(1, 2, 4), kernel_size: int = 3,
                 attn_tau: float = 1.5, res_alpha: float = 1.0, mix_beta: float = 1.0,
                 norm_type: str = 'bn', dropout: float = 0.0):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.scales = tuple(dilations)
        self.S = len(self.scales)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for d in self.scales:
            pad = d * (kernel_size // 2)
            self.convs.append(
                nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=pad, dilation=d, groups=channels)
            )
            self.bns.append(_make_norm(norm_type, channels))
        self.act = nn.GELU()
        # inter-scale attention head (shared across channels, no channel mixing)
        self.scale_attn = nn.Linear(self.S * 2, self.S)
        self.attn_tau = attn_tau
        self.res_alpha = res_alpha
        self.mix_beta = mix_beta
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x_bci):
        # x_bci: [B, C, T]
        ys = []
        for conv, bn in zip(self.convs, self.bns):
            y = conv(x_bci)
            y = self.act(bn(y))
            ys.append(y)  # [B, C, T]

        # Inter-scale fusion: compute weights from mean and std over time per scale
        means = torch.stack([y.mean(dim=-1) for y in ys], dim=2)     # [B, C, S]
        stds  = torch.stack([y.var(dim=-1, unbiased=False).clamp_min(1e-8).sqrt() for y in ys], dim=2)  # [B, C, S]
        feats = torch.cat([means, stds], dim=2)                      # [B, C, 2S]
        logits = self.scale_attn(feats)                              # [B, C, S]
        weights = torch.softmax(logits / self.attn_tau, dim=-1)      # [B, C, S]

        # Weighted sum across scales
        fused_attn = 0.0
        for k, y in enumerate(ys):
            w = weights[:, :, k].unsqueeze(-1)  # [B, C, 1]
            fused_attn = fused_attn + w * y

        # Average mix as a stable baseline
        fused_avg = sum(ys) / self.S

        fused = fused_avg + self.mix_beta * fused_attn               # [B, C, T]
        fused = self.drop(fused)
        out = x_bci + self.res_alpha * fused                         # residual to preserve original signal
        return out


class TemporalPyramidPooling(nn.Module):
    """Depthwise temporal smoothing with multiple window sizes (residual).
    Uses fixed average filters with reflect padding to keep length.
    """
    def __init__(self, channels: int, windows=(3, 7, 15), gamma: float = 0.5):
        super().__init__()
        self.channels = channels
        self.windows = tuple(int(w) for w in windows)
        self.gamma = gamma
        # register fixed averaging kernels as buffers
        self.kernels = nn.ParameterList()
        for k in self.windows:
            weight = torch.ones(channels, 1, k) / float(k)
            conv = nn.Conv1d(channels, channels, kernel_size=k, groups=channels, bias=False)
            conv.weight = nn.Parameter(weight, requires_grad=False)
            self.kernels.append(conv)

    def forward(self, x_bct):
        y_sum = 0.0
        for conv in self.kernels:
            k = conv.kernel_size[0]
            pad = (k - 1) // 2
            x_pad = F.pad(x_bct, (pad, pad), mode='reflect')
            y = conv(x_pad)
            y_sum = y_sum + y
        y_avg = y_sum / len(self.kernels)
        return x_bct + self.gamma * y_avg


class InterChannelTransformer(nn.Module):
    """Lightweight transformer over channel tokens (inverted view).

    Embeds time series per channel with Linear(T->d), runs TransformerEncoder over C tokens,
    and projects back to pred_len per channel.
    """
    def __init__(self, seq_len: int, pred_len: int, c_in: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.embed = nn.Linear(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=False, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model))
        self.projector = nn.Linear(d_model, pred_len)

    def forward(self, x_bct: torch.Tensor) -> torch.Tensor:
        # x_bct: [B, C, T]
        z = self.embed(x_bct)              # [B, C, d]
        z = z.transpose(0, 1)              # [C, B, d] for TransformerEncoder (S=C)
        z = self.encoder(z)                # [C, B, d]
        z = z.transpose(0, 1)              # [B, C, d]
        y = self.projector(z)              # [B, C, pred_len]
        return y

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, c_in: int):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.c_in = c_in

        # Non-linear Stream (replaced by ConvBackbone)
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.conv_backbone = ConvBackbone(c_in=c_in, seq_len=seq_len, pred_len=pred_len,
                                          patch_len=patch_len, stride=stride, padding_patch=padding_patch,
                                          n_layers=4, d_model=64, d_ff=256, dropout=0.1, head_dropout=0.1)

    # Linear Stream
    # MLP
    self.fc5 = nn.Linear(seq_len, pred_len * 4)
    self.avgpool1 = nn.AvgPool1d(kernel_size=2)
    self.ln1 = nn.LayerNorm(pred_len * 2)

    self.fc6 = nn.Linear(pred_len * 2, pred_len)
    self.avgpool2 = nn.AvgPool1d(kernel_size=2)
    self.ln2 = nn.LayerNorm(pred_len // 2)

    self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # Streams Concatination (now with 3 streams: s, t, and inter-channel transformer)
        self.pre_fuse_dropout = nn.Dropout(0.2)
        self.pre_fuse_norm = nn.LayerNorm(pred_len * 3)
        self.fc8 = nn.Linear(pred_len * 3, pred_len)

        # New: Inter-channel gate (does not mix values; scales per channel using cross-channel context)
        self.ch_gate = ChannelSEGate(c_in, hidden_ratio=4)

        # New: Multi-scale temporal preprocessing (inner- & inter-scale learning)
        self.ms_season = MultiScaleDWConv(c_in, dilations=(1, 2, 4, 8, 16), kernel_size=3,
                                          attn_tau=1.5, res_alpha=1.0, mix_beta=1.0,
                                          norm_type='bn', dropout=0.1)
        self.ms_trend = MultiScaleDWConv(c_in, dilations=(1, 2, 4, 8, 16, 24), kernel_size=5,
                                         attn_tau=1.7, res_alpha=1.0, mix_beta=0.8,
                                         norm_type='bn', dropout=0.1)
        # Optional: temporal pyramid smoothing residual for trend
        self.tpp_trend = TemporalPyramidPooling(c_in, windows=(5, 11, 21), gamma=0.4)

        # New: Inter-channel transformer stream (inverted transformer)
        self.ictr = InterChannelTransformer(seq_len=seq_len, pred_len=pred_len, c_in=c_in, d_model=64, n_heads=4, n_layers=1, dropout=0.2)
        self.ic_dropout = nn.Dropout(0.2)
        self.ic_gate = nn.Parameter(torch.tensor(0.0))  # sigmoid(0)=0.5 initial contribution

    def forward(self, s, t):
        # x: [Batch, Input, Channel]
        # s - seasonality
        # t - trend
        
        s = s.permute(0,2,1) # to [Batch, Channel, Input]
        t = t.permute(0,2,1) # to [Batch, Channel, Input]

        # Inter-channel gating (uses cross-channel info, preserves per-channel independence via scaling)
        gate = self.ch_gate(s)  # [B, C, 1]
        s = s * gate
        t = t * gate

        # Multi-scale temporal processing (inner- & inter-scale)
        s = self.ms_season(s)  # [B, C, I]
        t = self.ms_trend(t)   # [B, C, I]
        t = self.tpp_trend(t)  # residual pyramid smoothing for trend

        # Inter-channel transformer stream (uses combined features)
        u = 0.5 * (s + t)
        x_ic = self.ictr(u)     # [B, C, pred_len]
        x_ic = self.ic_dropout(x_ic) * torch.sigmoid(self.ic_gate)
        
        # Channel split for channel independence
        B = s.shape[0] # Batch size
        C = s.shape[1] # Channel size
        I = s.shape[2] # Input size
        s = torch.reshape(s, (B*C, I)) # [Batch and Channel, Input]
        t = torch.reshape(t, (B*C, I)) # [Batch and Channel, Input]

        # Non-linear Stream via ConvBackbone
        s = torch.reshape(s, (B, C, I))
        s = self.conv_backbone(s)              # [B, C, pred_len]
        s = torch.reshape(s, (B * C, self.pred_len))

        # Linear Stream
        # MLP
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)

        # Streams Concatination
        x_ic_flat = torch.reshape(x_ic, (B * C, self.pred_len))
        x = torch.cat((s, t, x_ic_flat), dim=1)
        x = self.pre_fuse_dropout(x)
        x = self.pre_fuse_norm(x)
        x = self.fc8(x)

        # Channel concatination
        x = torch.reshape(x, (B, C, self.pred_len)) # [Batch, Channel, Output]

        x = x.permute(0,2,1) # to [Batch, Output, Channel]

        return x