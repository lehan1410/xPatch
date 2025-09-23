import torch
import torch.nn as nn
import torch.nn.functional as F
from .enhanced_ema import EnhancedEMA, MultiScaleEMA, DualEMA
from .dema import DEMA
from .multiband import MULTIBAND  
from .emd import EMD
from .enhanced_modules import AdaptiveInstanceNorm, CrossChannelFusion

class EnhancedDECOMP(nn.Module):
    """
    Enhanced Decomposition module with multiple improvements:
    - Adaptive decomposition strategies
    - Cross-component attention
    - Anti-distribution shift normalization
    - Learnable component weights
    """
    def __init__(self, ma_type, alpha, beta, seq_len, enc_in, 
                 mb_k_small=7, mb_k_large=31, emd_imfs=2):
        super(EnhancedDECOMP, self).__init__()
        
        self.ma_type = ma_type
        self.seq_len = seq_len
        self.enc_in = enc_in
        
        # Enhanced decomposition strategies
        if ma_type == 'ema':
            self.decomp = MultiScaleEMA(alphas=[alpha * 0.5, alpha, alpha * 1.5], seq_len=seq_len)
        elif ma_type == 'dema':
            self.decomp = DEMA(alpha, beta)
        elif ma_type == 'dual_ema':
            self.decomp = DualEMA(alpha_trend=beta, alpha_seasonal=alpha, seq_len=seq_len)
        elif ma_type == 'multiband':
            self.decomp = MULTIBAND(mb_k_small, mb_k_large)
        elif ma_type == 'emd':
            self.decomp = EMD(emd_imfs)
        else:
            # Fallback to enhanced single EMA
            self.decomp = EnhancedEMA(alpha=alpha, learnable=True, adaptive=True, seq_len=seq_len)
        
        # Adaptive normalization for distribution shift
        self.adaptive_norm = AdaptiveInstanceNorm(enc_in)
        
        # Cross-component attention for better separation
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=seq_len, 
            num_heads=min(8, seq_len // 8),
            dropout=0.1,
            batch_first=True
        )
        
        # Learnable component importance weights
        self.seasonal_weight = nn.Parameter(torch.ones(1))
        self.trend_weight = nn.Parameter(torch.ones(1))
        
        # Component refinement networks
        self.seasonal_refiner = nn.Sequential(
            nn.Conv1d(enc_in, enc_in, kernel_size=3, padding=1, groups=enc_in),
            nn.BatchNorm1d(enc_in),
            nn.GELU(),
            nn.Conv1d(enc_in, enc_in, kernel_size=1),
            nn.Tanh()  # Bound seasonal component
        )
        
        self.trend_refiner = nn.Sequential(
            nn.Conv1d(enc_in, enc_in, kernel_size=7, padding=3, groups=enc_in),
            nn.BatchNorm1d(enc_in), 
            nn.GELU(),
            nn.Conv1d(enc_in, enc_in, kernel_size=1)
        )
        
        # Cross-channel fusion if multiple channels
        if enc_in > 1:
            self.cross_channel_fusion = CrossChannelFusion(enc_in, seq_len)
        
        # Residual component learning
        self.residual_projector = nn.Sequential(
            nn.Linear(seq_len, seq_len // 2),
            nn.GELU(),
            nn.Linear(seq_len // 2, seq_len)
        )
        
        # Adaptive combination weights
        self.combination_net = nn.Sequential(
            nn.Linear(seq_len * 2, seq_len),
            nn.GELU(),
            nn.Linear(seq_len, 3),  # [seasonal, trend, residual]
            nn.Softmax(dim=-1)
        )
        
    def apply_cross_attention(self, seasonal, trend):
        """Apply cross-attention between seasonal and trend components"""
        B, L, C = seasonal.shape
        
        # Reshape for attention: [B*C, L, 1] -> [B*C, L, L]
        seasonal_flat = seasonal.view(B*C, L).unsqueeze(-1).expand(-1, -1, L)
        trend_flat = trend.view(B*C, L).unsqueeze(-1).expand(-1, -1, L)
        
        # Cross-attention
        seasonal_attended, _ = self.cross_attention(seasonal_flat, trend_flat, trend_flat)
        trend_attended, _ = self.cross_attention(trend_flat, seasonal_flat, seasonal_flat)
        
        # Extract and reshape
        seasonal_refined = seasonal_attended[:, :, 0].view(B, L, C)
        trend_refined = trend_attended[:, :, 0].view(B, L, C)
        
        return seasonal_refined, trend_refined
    
    def refine_components(self, seasonal, trend):
        """Apply component-specific refinement"""
        # seasonal, trend: [B, L, C]
        B, L, C = seasonal.shape
        
        # Apply refinement networks
        seasonal_input = seasonal.permute(0, 2, 1)  # [B, C, L]
        trend_input = trend.permute(0, 2, 1)        # [B, C, L]
        
        seasonal_refined = self.seasonal_refiner(seasonal_input).permute(0, 2, 1)  # [B, L, C]
        trend_refined = self.trend_refiner(trend_input).permute(0, 2, 1)           # [B, L, C]
        
        # Apply learnable weights
        seasonal_weighted = seasonal + seasonal_refined * torch.sigmoid(self.seasonal_weight)
        trend_weighted = trend + trend_refined * torch.sigmoid(self.trend_weight)
        
        return seasonal_weighted, trend_weighted
    
    def adaptive_combination(self, original, seasonal, trend, residual=None):
        """Adaptively combine components based on input characteristics"""
        B, L, C = original.shape
        
        if residual is None:
            residual = torch.zeros_like(original)
        
        # Compute adaptive weights based on input statistics
        combined_features = torch.cat([seasonal, trend], dim=1)  # [B, 2*L, C]
        combined_flat = combined_features.view(B, -1)  # [B, 2*L*C]
        
        # Get combination weights for each sample
        weights = self.combination_net(combined_flat.mean(dim=-1, keepdim=True).expand(-1, 2*L))  # [B, 3]
        
        # Apply weights
        result = (weights[:, 0:1].unsqueeze(-1) * seasonal + 
                 weights[:, 1:2].unsqueeze(-1) * trend + 
                 weights[:, 2:3].unsqueeze(-1) * residual)
        
        return result, weights
    
    def forward(self, x):
        """
        Enhanced decomposition with multiple improvements
        x: [Batch, Input, Channel]
        """
        # Apply adaptive normalization
        x_norm = x.permute(0, 2, 1)  # [B, C, L]
        x_norm = self.adaptive_norm(x_norm)
        x_norm = x_norm.permute(0, 2, 1)  # [B, L, C]
        
        # Apply cross-channel fusion if available
        if hasattr(self, 'cross_channel_fusion'):
            x_fused = x_norm.permute(0, 2, 1)  # [B, C, L]
            x_fused = self.cross_channel_fusion(x_fused)
            x_norm = x_fused.permute(0, 2, 1)  # [B, L, C]
        
        # Apply decomposition based on type
        if self.ma_type == 'dual_ema':
            # Special handling for dual EMA
            result, trend, seasonal = self.decomp(x_norm)
            seasonal, trend = seasonal, trend  # Swap for consistency
        elif self.ma_type in ['dema', 'multiband', 'emd']:
            # These return (seasonal, trend)
            seasonal, trend = self.decomp(x_norm)
        else:
            # Single component decomposition (EMA variants)
            trend = self.decomp(x_norm)
            seasonal = x_norm - trend
        
        # Apply cross-attention refinement
        seasonal_refined, trend_refined = self.apply_cross_attention(seasonal, trend)
        
        # Apply component-specific refinement
        seasonal_final, trend_final = self.refine_components(seasonal_refined, trend_refined)
        
        # Compute residual component
        reconstructed = seasonal_final + trend_final
        residual = x_norm - reconstructed
        residual_refined = self.residual_projector(residual.view(-1, self.seq_len)).view_as(residual)
        
        # Adaptive combination for final output
        final_result, combination_weights = self.adaptive_combination(
            x_norm, seasonal_final, trend_final, residual_refined
        )
        
        # Ensure reconstruction quality
        seasonal_output = seasonal_final + 0.1 * (x_norm - reconstructed)
        trend_output = trend_final + 0.1 * (x_norm - reconstructed)
        
        return seasonal_output, trend_output

class DECOMP(nn.Module):
    """
    Original DECOMP - kept for backward compatibility with minimal enhancements
    """
    def __init__(self, ma_type, alpha, beta, seq_len, enc_in, 
                 mb_k_small=7, mb_k_large=31, emd_imfs=2):
        super(DECOMP, self).__init__()
        
        self.ma_type = ma_type
        
        if ma_type == 'ema':
            self.decomp = EnhancedEMA(alpha=alpha, learnable=True, adaptive=False)
        elif ma_type == 'dema':
            self.decomp = DEMA(alpha, beta)
        elif ma_type == 'multiband':
            self.decomp = MULTIBAND(mb_k_small, mb_k_large)
        elif ma_type == 'emd':
            self.decomp = EMD(emd_imfs)
        
        # Add basic normalization for stability
        self.norm = nn.LayerNorm(enc_in)
        
    def forward(self, x):
        # Apply normalization
        x = self.norm(x)
        
        if self.ma_type in ['dema', 'multiband', 'emd']:
            seasonal, trend = self.decomp(x)
        else:  # ema
            trend = self.decomp(x)
            seasonal = x - trend
            
        return seasonal, trend