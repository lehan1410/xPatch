import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AdaptiveInstanceNorm(nn.Module):
    """
    Adaptive Instance Normalization to handle distribution shift
    """
    def __init__(self, num_features, eps=1e-8):
        super(AdaptiveInstanceNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        # x: [B, C, L]
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight.unsqueeze(0).unsqueeze(-1) + self.bias.unsqueeze(0).unsqueeze(-1)

class AdaptiveMovingNorm(nn.Module):
    """
    Adaptive Moving Average Normalization for temporal adaptation
    """
    def __init__(self, num_features, momentum=0.1, eps=1e-8):
        super(AdaptiveMovingNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
    def forward(self, x):
        # x: [B, C, L]
        if self.training:
            mean = x.mean(dim=(0, -1))
            var = x.var(dim=(0, -1), unbiased=False)
            
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
                
            x_norm = (x - mean.unsqueeze(0).unsqueeze(-1)) / torch.sqrt(var.unsqueeze(0).unsqueeze(-1) + self.eps)
        else:
            x_norm = (x - self.running_mean.unsqueeze(0).unsqueeze(-1)) / torch.sqrt(self.running_var.unsqueeze(0).unsqueeze(-1) + self.eps)
            
        return x_norm * self.weight.unsqueeze(0).unsqueeze(-1) + self.bias.unsqueeze(0).unsqueeze(-1)

class HighwayConnection(nn.Module):
    """
    Highway Network for selective information flow
    """
    def __init__(self, dim):
        super(HighwayConnection, self).__init__()
        self.transform = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        H = self.relu(self.transform(x))
        T = torch.sigmoid(self.gate(x))
        return H * T + x * (1 - T)

class MultiScaleConv(nn.Module):
    """
    Multi-scale Temporal Convolutions for different temporal patterns
    """
    def __init__(self, in_channels, out_channels, scales=[1, 3, 5, 7]):
        super(MultiScaleConv, self).__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels // len(scales), 
                     kernel_size=scale, padding=scale//2, dilation=1)
            for scale in scales
        ])
        self.norm = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        out = torch.cat(outputs, dim=1)
        return self.norm(out)

class DilatedConvBlock(nn.Module):
    """
    Dilated Convolutions with exponential dilation for long-range dependencies
    """
    def __init__(self, channels, num_layers=4, kernel_size=3):
        super(DilatedConvBlock, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size=kernel_size, 
                     dilation=2**i, padding=(kernel_size-1)*(2**i)//2)
            for i in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.BatchNorm1d(channels) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residual = x
        for conv, norm in zip(self.convs, self.norms):
            x = F.gelu(norm(conv(x)))
            x = self.dropout(x)
        return x + residual

class ChannelAttention(nn.Module):
    """
    Channel Attention Mechanism for inter-channel dependencies
    """
    def __init__(self, num_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(num_channels, max(num_channels // reduction, 1)),
            nn.ReLU(),
            nn.Linear(max(num_channels // reduction, 1), num_channels)
        )
        
    def forward(self, x):
        # x: [B, C, L]
        avg_pool = self.avg_pool(x).squeeze(-1)  # [B, C]
        max_pool = self.max_pool(x).squeeze(-1)  # [B, C]
        
        avg_out = self.fc(avg_pool)
        max_out = self.fc(max_pool)
        
        attention = torch.sigmoid(avg_out + max_out).unsqueeze(-1)
        return x * attention

class SpatialAttention(nn.Module):
    """
    Spatial/Temporal Attention for important time points
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size//2, bias=False)
        
    def forward(self, x):
        # x: [B, C, L]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, L]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, L]
        
        concat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, L]
        attention = torch.sigmoid(self.conv(concat))  # [B, 1, L]
        
        return x * attention

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module combining Channel and Spatial attention
    """
    def __init__(self, num_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(num_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class CrossChannelFusion(nn.Module):
    """
    Cross-Channel Temporal Fusion for channel interaction
    """
    def __init__(self, num_channels, seq_len):
        super(CrossChannelFusion, self).__init__()
        self.channel_mixer = nn.Conv1d(num_channels, num_channels, 1, groups=1)
        self.temporal_mixer = nn.Conv1d(seq_len, seq_len, 1, groups=1)
        self.norm1 = nn.BatchNorm1d(num_channels)
        self.norm2 = nn.BatchNorm1d(seq_len)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x: [B, C, L]
        residual = x
        
        # Channel mixing
        x_mixed = F.gelu(self.norm1(self.channel_mixer(x)))
        x_mixed = self.dropout(x_mixed)
        
        # Temporal mixing across channels
        x_t = x.permute(0, 2, 1)  # [B, L, C]
        x_t_mixed = F.gelu(self.norm2(self.temporal_mixer(x_t)))
        x_t_mixed = self.dropout(x_t_mixed)
        x_t_mixed = x_t_mixed.permute(0, 2, 1)  # [B, C, L]
        
        return residual + x_mixed + x_t_mixed

class LowRankLinear(nn.Module):
    """
    Low-rank approximation for parameter efficiency
    """
    def __init__(self, in_features, out_features, rank=None):
        super(LowRankLinear, self).__init__()
        if rank is None:
            rank = min(in_features, out_features) // 4
        
        self.rank = rank
        self.left = nn.Linear(in_features, rank, bias=False)
        self.right = nn.Linear(rank, out_features, bias=True)
        
    def forward(self, x):
        return self.right(self.left(x))

class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit for enhanced non-linearity
    """
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super(GatedLinearUnit, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear(x)
        x, gate = x.chunk(2, dim=-1)
        return x * torch.sigmoid(gate)

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer-like architectures
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        return x + self.pe[:x.size(0), :]

class SharedChannelProcessor(nn.Module):
    """
    Parameter sharing across channels for scalability
    """
    def __init__(self, seq_len, pred_len, num_channels):
        super(SharedChannelProcessor, self).__init__()
        # Shared parameters for all channels
        self.shared_linear = nn.Linear(seq_len, pred_len)
        # Channel-specific scaling factors
        self.channel_scales = nn.Parameter(torch.ones(num_channels, 1, 1))
        self.channel_bias = nn.Parameter(torch.zeros(num_channels, 1, 1))
        
    def forward(self, x):
        # x: [B, C, L]
        shared_out = self.shared_linear(x)
        return shared_out * self.channel_scales + self.channel_bias

class AdaptivePooling(nn.Module):
    """
    Adaptive pooling with learnable pooling strategies
    """
    def __init__(self, seq_len, target_len):
        super(AdaptivePooling, self).__init__()
        self.seq_len = seq_len
        self.target_len = target_len
        self.pool_weight = nn.Parameter(torch.ones(3))  # avg, max, adaptive_avg
        
    def forward(self, x):
        # x: [B, C, L]
        weights = F.softmax(self.pool_weight, dim=0)
        
        # Different pooling strategies
        avg_pool = F.avg_pool1d(x, kernel_size=self.seq_len // self.target_len)
        max_pool = F.max_pool1d(x, kernel_size=self.seq_len // self.target_len)
        adaptive_pool = F.adaptive_avg_pool1d(x, self.target_len)
        
        # Weighted combination
        out = (weights[0] * avg_pool + 
               weights[1] * max_pool + 
               weights[2] * adaptive_pool)
        
        return out