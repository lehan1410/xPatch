import torch
from torch import nn
import torch.nn.functional as F

class EnhancedEMA(nn.Module):
    """
    Enhanced Exponential Moving Average with adaptive alpha and anti-distribution shift
    """
    def __init__(self, alpha=0.3, learnable=True, adaptive=True, seq_len=None):
        super(EnhancedEMA, self).__init__()
        self.learnable = learnable
        self.adaptive = adaptive
        self.seq_len = seq_len
        
        if learnable:
            # Learnable alpha parameter
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = alpha
            
        if adaptive and seq_len is not None:
            # Adaptive alpha based on temporal position
            self.position_alpha = nn.Parameter(torch.ones(seq_len) * alpha)
            # Temporal attention for adaptive weighting
            self.temporal_attention = nn.Sequential(
                nn.Linear(seq_len, seq_len // 4),
                nn.ReLU(),
                nn.Linear(seq_len // 4, seq_len),
                nn.Softmax(dim=-1)
            )
        
        # Anti-distribution shift components
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))
        self.momentum = 0.1
        
        # Learnable temperature for smoothing
        self.temperature = nn.Parameter(torch.ones(1))
        
    def clamp_alpha(self):
        """Ensure alpha stays in valid range [0, 1]"""
        if self.learnable:
            self.alpha.data.clamp_(0.01, 0.99)
        if hasattr(self, 'position_alpha'):
            self.position_alpha.data.clamp_(0.01, 0.99)
    
    def adaptive_forward(self, x):
        """Forward pass with adaptive alpha"""
        # x: [Batch, Input, Channel]
        B, T, C = x.shape
        
        # Get temporal attention weights
        x_flat = x.view(B, -1)  # [B, T*C]
        temp_weights = self.temporal_attention(x_flat.mean(dim=-1, keepdim=True).expand(-1, T))
        
        # Adaptive alpha based on position and attention
        adaptive_alpha = self.position_alpha.unsqueeze(0).unsqueeze(-1) * temp_weights.unsqueeze(-1)
        
        # Initialize with first timestep
        s = x[:, 0, :].unsqueeze(1)  # [B, 1, C]
        results = [s]
        
        for t in range(1, T):
            xt = x[:, t, :].unsqueeze(1)  # [B, 1, C]
            alpha_t = adaptive_alpha[:, t, :].unsqueeze(1)  # [B, 1, C]
            s = alpha_t * xt + (1 - alpha_t) * s
            results.append(s)
            
        return torch.cat(results, dim=1)
    
    def optimized_forward(self, x):
        """Optimized implementation with O(1) time complexity"""
        # x: [Batch, Input, Channel]
        B, T, C = x.shape
        
        # Clamp alpha to valid range
        self.clamp_alpha()
        
        if self.learnable:
            alpha = self.alpha
        else:
            alpha = self.alpha
            
        # Apply temperature scaling
        alpha = alpha * torch.sigmoid(self.temperature)
        
        # Create exponential weights
        powers = torch.flip(torch.arange(T, dtype=torch.float32, device=x.device), dims=(0,))
        weights = torch.pow((1 - alpha), powers)
        divisor = weights.clone()
        weights[1:] = weights[1:] * alpha
        
        # Reshape for broadcasting
        weights = weights.view(1, T, 1).expand(B, T, C)
        divisor = divisor.view(1, T, 1).expand(B, T, C)
        
        # Compute EMA with cumulative sum
        x_weighted = x * weights
        x_cumsum = torch.cumsum(x_weighted, dim=1)
        result = x_cumsum / divisor
        
        # Anti-distribution shift: normalize output
        if self.training:
            batch_mean = result.mean()
            batch_var = result.var()
            
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        
        # Apply running statistics for stability
        result = (result - self.running_mean) / torch.sqrt(self.running_var + 1e-8)
        
        return result
    
    def forward(self, x):
        """Main forward function"""
        if self.adaptive and hasattr(self, 'position_alpha'):
            return self.adaptive_forward(x)
        else:
            return self.optimized_forward(x)

class MultiScaleEMA(nn.Module):
    """
    Multi-scale EMA for capturing different temporal patterns
    """
    def __init__(self, alphas=[0.1, 0.3, 0.7], seq_len=None):
        super(MultiScaleEMA, self).__init__()
        self.num_scales = len(alphas)
        self.emas = nn.ModuleList([
            EnhancedEMA(alpha=alpha, learnable=True, adaptive=True, seq_len=seq_len)
            for alpha in alphas
        ])
        
        # Learnable weights for combining scales
        self.scale_weights = nn.Parameter(torch.ones(self.num_scales))
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.num_scales, self.num_scales * 2),
            nn.ReLU(),
            nn.Linear(self.num_scales * 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Apply EMA at different scales
        ema_outputs = []
        for ema in self.emas:
            ema_out = ema(x)
            ema_outputs.append(ema_out)
        
        # Stack outputs: [B, T, C, num_scales]
        stacked = torch.stack(ema_outputs, dim=-1)
        
        # Normalize scale weights
        weights = F.softmax(self.scale_weights, dim=0)
        
        # Weighted combination
        weighted = stacked * weights.view(1, 1, 1, -1)
        
        # Adaptive fusion based on input statistics
        input_stats = x.std(dim=1, keepdim=True)  # [B, 1, C]
        fusion_weights = self.fusion(input_stats.expand(-1, x.size(1), -1))  # [B, T, C]
        
        # Final combination
        result = weighted.sum(dim=-1)  # [B, T, C]
        result = result * fusion_weights + x * (1 - fusion_weights)
        
        return result

class DualEMA(nn.Module):
    """
    Dual EMA for trend and seasonality with different characteristics
    """
    def __init__(self, alpha_trend=0.7, alpha_seasonal=0.3, seq_len=None):
        super(DualEMA, self).__init__()
        
        # Trend EMA (slower, captures long-term patterns)
        self.trend_ema = EnhancedEMA(
            alpha=alpha_trend, 
            learnable=True, 
            adaptive=True, 
            seq_len=seq_len
        )
        
        # Seasonal EMA (faster, captures short-term patterns)
        self.seasonal_ema = EnhancedEMA(
            alpha=alpha_seasonal, 
            learnable=True, 
            adaptive=True, 
            seq_len=seq_len
        )
        
        # Learnable combination weights
        self.combination_weight = nn.Parameter(torch.tensor(0.5))
        
        # Cross-component attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=seq_len, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, x):
        # Apply dual EMAs
        trend = self.trend_ema(x)
        seasonal = self.seasonal_ema(x)
        
        # Cross-attention between trend and seasonal
        # Reshape for attention: [B*C, T, 1] -> [B*C, T, embed_dim]
        B, T, C = x.shape
        
        trend_flat = trend.view(B*C, T).unsqueeze(-1).expand(-1, -1, T)
        seasonal_flat = seasonal.view(B*C, T).unsqueeze(-1).expand(-1, -1, T)
        
        # Apply cross-attention
        trend_attended, _ = self.cross_attention(trend_flat, seasonal_flat, seasonal_flat)
        seasonal_attended, _ = self.cross_attention(seasonal_flat, trend_flat, trend_flat)
        
        # Extract and reshape
        trend_attended = trend_attended[:, :, 0].view(B, T, C)
        seasonal_attended = seasonal_attended[:, :, 0].view(B, T, C)
        
        # Learnable combination
        weight = torch.sigmoid(self.combination_weight)
        result = weight * trend_attended + (1 - weight) * seasonal_attended
        
        return result, trend, seasonal

class EMA(nn.Module):
    """
    Original EMA block - kept for backward compatibility
    """
    def __init__(self, alpha):
        super(EMA, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        # x: [Batch, Input, Channel]
        _, t, _ = x.shape
        powers = torch.flip(torch.arange(t, dtype=torch.double, device=x.device), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers)
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)