import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .enhanced_modules import (
    PositionalEncoding, HighwayConnection, MultiScaleConv, 
    DilatedConvBlock, CBAM, CrossChannelFusion, LowRankLinear,
    GatedLinearUnit, AdaptiveInstanceNorm
)

class EnhancedMultiHeadAttention(nn.Module):
    """
    Enhanced Multi-Head Attention with relative position encoding and improved long-range modeling
    """
    def __init__(self, d_model, nhead, dropout=0.1, max_relative_position=32):
        super(EnhancedMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        assert self.head_dim * nhead == d_model, "d_model must be divisible by nhead"
        
        # Standard attention components
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        # Relative position encoding
        self.max_relative_position = max_relative_position
        self.relative_position_k = nn.Embedding(2 * max_relative_position + 1, self.head_dim)
        self.relative_position_v = nn.Embedding(2 * max_relative_position + 1, self.head_dim)
        
        # Long-range attention enhancement
        self.global_attention = nn.MultiheadAttention(d_model, nhead//2, dropout=dropout, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def get_relative_positions(self, seq_len):
        """Generate relative position matrix"""
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.unsqueeze(0).expand(seq_len, seq_len)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        distance_mat = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        distance_mat = distance_mat + self.max_relative_position
        return distance_mat
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.size()
        
        # Standard self-attention
        q = self.q_linear(query).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Add relative position encoding
        if seq_len <= self.max_relative_position * 2:
            rel_pos_ids = self.get_relative_positions(seq_len).to(query.device)
            rel_pos_k = self.relative_position_k(rel_pos_ids)
            rel_pos_v = self.relative_position_v(rel_pos_ids)
            
            # Add relative position bias to attention scores
            rel_scores = torch.einsum('bhld,lrd->bhlr', q, rel_pos_k)
            scores = scores + rel_scores
        
        # Apply mask and softmax
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        local_attn = torch.matmul(attn_weights, v)
        
        # Global attention for long-range dependencies
        query_global = query.view(batch_size, seq_len, d_model)
        global_attn, _ = self.global_attention(query_global, key, value)
        
        # Combine local and global attention
        local_attn = local_attn.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        combined_attn = 0.7 * local_attn + 0.3 * global_attn
        
        return self.out_linear(combined_attn)

class EnhancedTransformerLayer(nn.Module):
    """
    Enhanced Transformer Layer with highway connections and improved feed-forward
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(EnhancedTransformerLayer, self).__init__()
        
        # Enhanced attention
        self.self_attn = EnhancedMultiHeadAttention(d_model, nhead, dropout)
        
        # Enhanced feed-forward with GLU and highway connections
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, d_model)
        )
        
        # Highway connections
        self.highway1 = HighwayConnection(d_model)
        self.highway2 = HighwayConnection(d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Gated Linear Unit for better non-linearity
        self.glu = GatedLinearUnit(d_model, d_model, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Multi-head attention with highway connection
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.highway1(attn_out))
        
        # Enhanced feed-forward with highway connection
        ff_out = self.feed_forward(x)
        ff_out = self.glu(ff_out)
        x = self.norm2(x + self.highway2(ff_out))
        
        return x

class AdaptivePatching(nn.Module):
    """
    Adaptive patching with learnable patch sizes and overlapping
    """
    def __init__(self, seq_len, patch_len, stride, d_model):
        super(AdaptivePatching, self).__init__()
        self.seq_len = seq_len
        self.base_patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # Learnable patch size variations
        self.patch_variations = nn.Parameter(torch.tensor([0.8, 1.0, 1.2]))
        
        # Multi-scale patch embedding
        self.patch_embeddings = nn.ModuleList([
            nn.Linear(int(patch_len * var), d_model) 
            for var in [0.8, 1.0, 1.2]
        ])
        
        # Attention for patch combination
        self.patch_attention = nn.MultiheadAttention(d_model, 4, batch_first=True)
        
    def forward(self, x):
        # x: [B*C, seq_len]
        batch_size, seq_len = x.shape
        
        # Create patches with different sizes
        patch_outputs = []
        
        for i, var in enumerate([0.8, 1.0, 1.2]):
            patch_len = max(1, int(self.base_patch_len * var))
            if seq_len >= patch_len:
                # Create patches
                patches = x.unfold(-1, patch_len, self.stride)  # [B*C, num_patches, patch_len]
                # Embed patches
                embedded = self.patch_embeddings[i](patches)  # [B*C, num_patches, d_model]
                patch_outputs.append(embedded)
        
        if not patch_outputs:
            # Fallback for very short sequences
            patches = x.unsqueeze(-1).expand(-1, -1, self.d_model)
            return patches.unsqueeze(1)  # [B*C, 1, d_model]
        
        # Combine different patch scales using attention
        if len(patch_outputs) > 1:
            # Pad to same length
            max_patches = max(p.size(1) for p in patch_outputs)
            padded_patches = []
            
            for patches in patch_outputs:
                if patches.size(1) < max_patches:
                    padding = torch.zeros(batch_size, max_patches - patches.size(1), self.d_model, 
                                        device=patches.device)
                    patches = torch.cat([patches, padding], dim=1)
                padded_patches.append(patches)
            
            # Stack and apply attention
            stacked = torch.stack(padded_patches, dim=0)  # [num_scales, B*C, max_patches, d_model]
            stacked = stacked.view(-1, max_patches, self.d_model)  # [num_scales*B*C, max_patches, d_model]
            
            attended, _ = self.patch_attention(stacked, stacked, stacked)
            
            # Reshape and average across scales
            attended = attended.view(len(patch_outputs), batch_size, max_patches, self.d_model)
            result = attended.mean(dim=0)  # [B*C, max_patches, d_model]
        else:
            result = patch_outputs[0]
        
        return result

class EnhancedTransformerNetwork(nn.Module):
    """
    Enhanced Transformer Network with all improvements
    """
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, 
                 d_model=512, nhead=8, num_layers=3, dropout=0.1, num_channels=1):
        super(EnhancedTransformerNetwork, self).__init__()
        
        # Parameters
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.num_channels = num_channels
        
        # Enhanced patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.patch_num = (seq_len - patch_len) // stride + 1
        
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1
        
        # Adaptive patching
        self.adaptive_patching = AdaptivePatching(seq_len, patch_len, stride, d_model)
        
        # Enhanced normalization
        self.adaptive_norm = AdaptiveInstanceNorm(num_channels)
        
        # Channel processing
        self.cbam = CBAM(num_channels) if num_channels > 1 else None
        self.cross_channel_fusion = CrossChannelFusion(num_channels, seq_len) if num_channels > 1 else None
        
        # Enhanced positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max(seq_len, self.patch_num))
        
        # Seasonal processing (more complex patterns)
        self.seasonal_layers = nn.ModuleList([
            EnhancedTransformerLayer(d_model, nhead, 4*d_model, dropout)
            for _ in range(num_layers)
        ])
        
        # Trend processing (simpler, fewer layers)
        trend_layers = max(1, num_layers // 2)
        self.trend_layers = nn.ModuleList([
            EnhancedTransformerLayer(d_model, max(nhead//2, 1), 2*d_model, dropout)
            for _ in range(trend_layers)
        ])
        
        # Multi-scale convolutions for local patterns
        self.multiscale_conv = MultiScaleConv(d_model, d_model, scales=[1, 3, 5])
        
        # Dilated convolutions for long-range dependencies
        self.dilated_conv = DilatedConvBlock(d_model, num_layers=3)
        
        # Enhanced decoders
        self.seasonal_decoder = nn.Sequential(
            LowRankLinear(d_model, pred_len * 2, rank=pred_len),
            nn.GELU(),
            nn.Dropout(dropout),
            LowRankLinear(pred_len * 2, pred_len, rank=pred_len // 2)
        )
        
        self.trend_decoder = nn.Sequential(
            LowRankLinear(d_model, pred_len * 2, rank=pred_len),
            nn.GELU(),
            nn.Dropout(dropout),
            LowRankLinear(pred_len * 2, pred_len, rank=pred_len // 2)
        )
        
        # Adaptive fusion with learnable gating
        self.fusion_gate = nn.Sequential(
            nn.Linear(pred_len * 2, pred_len),
            nn.Sigmoid()
        )
        
        # Final projection with highway connection
        self.final_projection = LowRankLinear(pred_len * 2, pred_len, rank=pred_len // 2)
        self.final_highway = HighwayConnection(pred_len)
        
        # Learnable stream importance
        self.stream_importance = nn.Parameter(torch.tensor([0.6, 0.4]))  # [seasonal, trend]
        
    def process_channel_dependencies(self, x):
        """Process inter-channel dependencies if multiple channels"""
        if self.num_channels > 1 and self.cbam is not None:
            x = self.adaptive_norm(x)
            x = self.cbam(x)
            if self.cross_channel_fusion is not None:
                x = self.cross_channel_fusion(x)
        return x
        
    def enhanced_patching(self, x):
        """Apply enhanced adaptive patching"""
        B, C, I = x.shape
        x = x.reshape(B * C, I)
        
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
            
        # Use adaptive patching
        patches = self.adaptive_patching(x)  # [B*C, num_patches, d_model]
        
        return patches, B, C
    
    def process_transformer_branch(self, x, layers, use_conv_enhancement=True):
        """Process through transformer layers with optional conv enhancement"""
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer layers
        for layer in layers:
            x = layer(x)
        
        # Optional convolution enhancement for local patterns
        if use_conv_enhancement and x.size(1) > 1:
            # x: [B*C, seq, d_model] -> [B*C, d_model, seq]
            x_conv = x.transpose(1, 2)
            
            # Apply multi-scale and dilated convolutions
            x_multi = self.multiscale_conv(x_conv)
            x_dilated = self.dilated_conv(x_conv)
            
            # Combine with residual connection
            x_enhanced = x_conv + x_multi + x_dilated
            x = x_enhanced.transpose(1, 2)  # Back to [B*C, seq, d_model]
        
        return x
    
    def forward(self, s, t):
        # s, t: [Batch, Input, Channel]
        B, I, C = s.shape
        
        # Process channel dependencies
        s = s.permute(0, 2, 1)  # [B, C, I]
        t = t.permute(0, 2, 1)  # [B, C, I]
        
        s = self.process_channel_dependencies(s)
        t = self.process_channel_dependencies(t)
        
        # Enhanced patching
        s_patches, Bs, Cs = self.enhanced_patching(s)  # [B*C, num_patches, d_model]
        t_patches, Bt, Ct = self.enhanced_patching(t)  # [B*C, num_patches, d_model]
        
        # Process seasonal branch (more complex)
        s_encoded = self.process_transformer_branch(s_patches, self.seasonal_layers, use_conv_enhancement=True)
        
        # Process trend branch (simpler)
        t_encoded = self.process_transformer_branch(t_patches, self.trend_layers, use_conv_enhancement=False)
        
        # Global pooling and decoding
        s_pooled = s_encoded.mean(dim=1)  # [B*C, d_model]
        t_pooled = t_encoded.mean(dim=1)  # [B*C, d_model]
        
        # Decode to predictions
        s_pred = self.seasonal_decoder(s_pooled)  # [B*C, pred_len]
        t_pred = self.trend_decoder(t_pooled)    # [B*C, pred_len]
        
        # Reshape to [B, C, pred_len]
        s_pred = s_pred.view(B, C, self.pred_len)
        t_pred = t_pred.view(B, C, self.pred_len)
        
        # Adaptive fusion with learnable importance
        importance = F.softmax(self.stream_importance, dim=0)
        
        # Concatenate for fusion gate
        combined = torch.cat([s_pred, t_pred], dim=-1)  # [B, C, pred_len*2]
        gate = self.fusion_gate(combined)  # [B, C, pred_len]
        
        # Weighted combination
        fused = (importance[0] * gate * s_pred + 
                importance[1] * (1 - gate) * t_pred)
        
        # Final projection with highway connection
        concat_features = torch.cat([s_pred, t_pred], dim=-1)  # [B, C, pred_len*2]
        final_out = self.final_projection(concat_features)  # [B, C, pred_len]
        final_out = self.final_highway(final_out)
        
        # Combine fused and final outputs
        output = 0.7 * fused + 0.3 * final_out
        
        # Reshape to [B, pred_len, C]
        output = output.permute(0, 2, 1)
        
        return output

# Backward compatibility - keep original transformer
class TransformerNetwork(nn.Module):
    """Original TransformerNetwork with minimal enhancements"""
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, 
                 d_model=512, nhead=8, num_layers=3, dropout=0.1):
        super(TransformerNetwork, self).__init__()
        
        # Parameters
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        # Patching params
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1
        else:
            self.padding_patch_layer = None
        
        # Input projection with highway connection
        self.input_projection = nn.Linear(self.patch_len, d_model)
        self.input_highway = HighwayConnection(d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Enhanced Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=4*d_model, dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Enhanced Decoder
        self.decoder = nn.Sequential(
            LowRankLinear(d_model * self.patch_num, pred_len * 2, rank=pred_len),
            nn.GELU(),
            nn.Dropout(dropout),
            LowRankLinear(pred_len * 2, pred_len, rank=pred_len // 2)
        )

        # Trend processing
        self.t_input_projection = nn.Linear(self.patch_len, d_model)
        self.t_input_highway = HighwayConnection(d_model)
        
        t_layers = max(1, num_layers // 2)
        t_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=4*d_model, dropout=dropout,
            activation='gelu'
        )
        self.t_transformer_encoder = nn.TransformerEncoder(t_enc_layer, num_layers=t_layers)
        
        self.t_decoder = nn.Sequential(
            LowRankLinear(d_model * self.patch_num, pred_len * 2, rank=pred_len),
            nn.GELU(),
            nn.Dropout(dropout),
            LowRankLinear(pred_len * 2, pred_len, rank=pred_len // 2)
        )
        
        # Enhanced final fusion
        self.fc8 = LowRankLinear(pred_len * 2, pred_len, rank=pred_len // 2)
        self.gate_fc = nn.Sequential(
            nn.Linear(2, 4),
            nn.GELU(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
    
    def _patch(self, x):
        # x: [B, C, seq_len] -> [B*C, patch_num, patch_len]
        B, C, I = x.shape
        x = x.reshape(B * C, I)
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return x, B, C

    def _season_branch(self, s):
        # s: [B, I, C] -> [B, pred_len, C]
        B, I, C = s.shape
        s = s.permute(0, 2, 1)                  # [B, C, I]
        s, Bc, Cc = self._patch(s)               # [B*C, P, Lp]
        s = self.input_projection(s)            # [B*C, P, d_model]
        s = self.input_highway(s)               # Enhanced with highway
        s = self.pos_encoder(s)                 # [B*C, P, d_model]
        s = s.permute(1, 0, 2)                  # [P, B*C, d_model]
        s = self.transformer_encoder(s)
        s = s.permute(1, 0, 2).reshape(B * C, -1)  # [B*C, P*d_model]
        s = self.decoder(s).view(B, C, self.pred_len).permute(0, 2, 1)
        return s

    def _trend_branch_attn(self, t):
        # t: [B, I, C] -> [B, pred_len, C]
        B, I, C = t.shape
        t = t.permute(0, 2, 1)                  # [B, C, I]
        t, Bc, Cc = self._patch(t)               # [B*C, P, Lp]
        t = self.t_input_projection(t)          # [B*C, P, d_model]
        t = self.t_input_highway(t)             # Enhanced with highway
        t = self.pos_encoder(t)                 # [B*C, P, d_model]
        t = t.permute(1, 0, 2)                  # [P, B*C, d_model]
        t = self.t_transformer_encoder(t)
        t = t.permute(1, 0, 2).reshape(B * C, -1)  # [B*C, P*d_model]
        t = self.t_decoder(t).view(B, C, self.pred_len).permute(0, 2, 1)
        return t

    def forward(self, s, t):
        # s, t: [Batch, Input, Channel]
        B, I, C = s.shape

        s = self._season_branch(s)
        t = self._trend_branch_attn(t)

        # Enhanced gating
        g_in = torch.stack([s, t], dim=-1)              # [B, pred_len, C, 2]
        g = self.gate_fc(g_in).squeeze(-1)              # [B, pred_len, C]
        s_w = g * s
        t_w = (1.0 - g) * t

        # Enhanced projection head
        x = torch.cat([s_w, t_w], dim=1)                # [B, 2*pred_len, C]
        x = x.permute(0, 2, 1)                          # [B, C, 2*pred_len]
        x = self.fc8(x).permute(0, 2, 1)                # [B, pred_len, C]
        return x