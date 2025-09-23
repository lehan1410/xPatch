import torch
from torch import nn
import torch.nn.functional as F
from .enhanced_modules import (
    AdaptiveInstanceNorm, AdaptiveMovingNorm, HighwayConnection,
    MultiScaleConv, DilatedConvBlock, CBAM, CrossChannelFusion,
    LowRankLinear, GatedLinearUnit, SharedChannelProcessor, AdaptivePooling
)

class EnhancedNetwork(nn.Module):
    """
    Enhanced Network with all improvements for addressing linear predictor limitations
    """
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, num_channels=None):
        super(EnhancedNetwork, self).__init__()

        # Parameters
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.num_channels = num_channels if num_channels is not None else 1

        # Enhanced patching parameters
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1
        
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        # =================== NORMALIZATION IMPROVEMENTS ===================
        # Anti-distribution shift normalization
        self.adaptive_norm_s = AdaptiveInstanceNorm(self.num_channels)
        self.adaptive_norm_t = AdaptiveInstanceNorm(self.num_channels)
        self.moving_norm = AdaptiveMovingNorm(self.num_channels)

        # =================== CROSS-CHANNEL DEPENDENCIES ===================
        # Channel attention and cross-channel fusion
        self.cbam_s = CBAM(self.num_channels, reduction=max(self.num_channels // 16, 1))
        self.cbam_t = CBAM(self.num_channels, reduction=max(self.num_channels // 16, 1))
        self.cross_channel_fusion = CrossChannelFusion(self.num_channels, seq_len)

        # =================== NON-LINEAR STREAM ENHANCEMENTS ===================
        # Enhanced patch embedding with highway connections
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.highway1 = HighwayConnection(self.dim)
        self.glu1 = GatedLinearUnit(self.dim, self.dim)
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # Multi-scale convolutions for different temporal patterns
        self.multiscale_conv = MultiScaleConv(self.patch_num, self.patch_num, scales=[1, 3, 5, 7])
        
        # Dilated convolutions for long-range dependencies
        self.dilated_conv = DilatedConvBlock(self.patch_num, num_layers=4)

        # Enhanced CNN layers
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num, patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Enhanced residual stream with highway connection
        self.fc2 = LowRankLinear(self.dim, patch_len, rank=min(self.dim, patch_len) // 4)
        self.highway2 = HighwayConnection(patch_len)

        # Enhanced pointwise convolution
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Enhanced flatten head with better capacity
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = LowRankLinear(self.patch_num * patch_len, pred_len * 4, rank=pred_len)
        self.glu3 = GatedLinearUnit(pred_len * 4, pred_len * 2)
        self.fc4 = nn.Linear(pred_len * 2, pred_len)
        self.dropout1 = nn.Dropout(0.1)

        # =================== LINEAR STREAM ENHANCEMENTS ===================
        # Enhanced MLP with better non-linearity
        if self.num_channels > 16:  # Use shared parameters for scalability
            self.linear_processor = SharedChannelProcessor(seq_len, pred_len, self.num_channels)
        else:
            self.fc5 = LowRankLinear(seq_len, pred_len * 4, rank=pred_len)
            self.adaptive_pool1 = AdaptivePooling(pred_len * 4, pred_len * 2)
            self.ln1 = nn.LayerNorm(pred_len * 2)
            self.dropout2 = nn.Dropout(0.1)

            self.fc6 = LowRankLinear(pred_len * 2, pred_len, rank=pred_len // 2)
            self.adaptive_pool2 = AdaptivePooling(pred_len, pred_len // 2)
            self.ln2 = nn.LayerNorm(pred_len // 2)
            self.dropout3 = nn.Dropout(0.1)

            self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # =================== ENHANCED FUSION ===================
        # Better stream concatenation with attention
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=pred_len, 
            num_heads=min(8, pred_len // 8), 
            dropout=0.1,
            batch_first=True
        )
        self.fc8 = LowRankLinear(pred_len * 2, pred_len, rank=pred_len // 2)
        self.final_highway = HighwayConnection(pred_len)

        # =================== LEARNABLE COMPONENTS ===================
        # Learnable stream weights
        self.stream_weights = nn.Parameter(torch.ones(2))  # [non-linear, linear]
        
        # Temporal consistency components
        self.consistency_weight = nn.Parameter(torch.tensor(0.1))
        
        # Channel mixing weights for scalability
        if self.num_channels > 1:
            self.channel_mixer = nn.Parameter(torch.eye(self.num_channels) * 0.8 + 
                                            torch.ones(self.num_channels, self.num_channels) * 0.2)

    def apply_cross_channel_processing(self, x):
        """Apply cross-channel information fusion"""
        if self.num_channels > 1:
            # x: [B, C, L]
            x_mixed = torch.matmul(self.channel_mixer, x)
            x = self.cross_channel_fusion(x + x_mixed)
        return x

    def forward(self, s, t):
        # x: [Batch, Input, Channel]
        # s - seasonality, t - trend
        
        # =================== INPUT PREPROCESSING ===================
        s = s.permute(0, 2, 1)  # to [Batch, Channel, Input]
        t = t.permute(0, 2, 1)  # to [Batch, Channel, Input]
        
        B, C, I = s.shape
        
        # Apply adaptive normalization for distribution shift
        s = self.adaptive_norm_s(s)
        t = self.adaptive_norm_t(t)
        
        # Apply channel attention
        s = self.cbam_s(s)
        t = self.cbam_t(t)
        
        # Cross-channel processing
        s = self.apply_cross_channel_processing(s)
        t = self.apply_cross_channel_processing(t)
        
        # Channel split for processing
        s = torch.reshape(s, (B*C, I))  # [Batch and Channel, Input]
        t = torch.reshape(t, (B*C, I))  # [Batch and Channel, Input]

        # =================== NON-LINEAR STREAM ===================
        # Enhanced patching
        if self.padding_patch == 'end':
            s_padded = self.padding_patch_layer(s)
        else:
            s_padded = s
            
        s_patches = s_padded.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s_patches: [Batch and Channel, Patch_num, Patch_len]
        
        # Enhanced patch embedding
        s_emb = self.fc1(s_patches)
        s_emb = self.highway1(s_emb)
        s_emb = self.glu1(s_emb)
        s_emb = self.bn1(s_emb)
        
        # Multi-scale and dilated convolutions
        s_multiscale = self.multiscale_conv(s_emb)
        s_dilated = self.dilated_conv(s_emb)
        s_combined = s_emb + s_multiscale + s_dilated
        
        # Store residual
        res = s_combined
        
        # Enhanced CNN depthwise
        s_conv = self.conv1(s_combined)
        s_conv = self.gelu2(s_conv)
        s_conv = self.bn2(s_conv)
        
        # Enhanced residual connection
        res_transformed = self.fc2(res)
        res_transformed = self.highway2(res_transformed)
        s_conv = s_conv + res_transformed
        
        # Enhanced pointwise convolution
        s_final = self.conv2(s_conv)
        s_final = self.gelu3(s_final)
        s_final = self.bn3(s_final)
        
        # Enhanced flatten head
        s_flat = self.flatten1(s_final)
        s_out = self.fc3(s_flat)
        s_out = self.glu3(s_out)
        s_out = self.dropout1(s_out)
        s_out = self.fc4(s_out)

        # =================== LINEAR STREAM ===================
        if hasattr(self, 'linear_processor'):
            # Use shared processor for scalability
            t_reshaped = t.view(B, C, I)
            t_out = self.linear_processor(t_reshaped)
            t_out = t_out.view(B*C, self.pred_len)
        else:
            # Enhanced MLP
            t_mlp = self.fc5(t)
            t_mlp = self.adaptive_pool1(t_mlp.unsqueeze(1)).squeeze(1)
            t_mlp = self.ln1(t_mlp)
            t_mlp = self.dropout2(t_mlp)
            
            t_mlp = self.fc6(t_mlp)
            t_mlp = self.adaptive_pool2(t_mlp.unsqueeze(1)).squeeze(1)
            t_mlp = self.ln2(t_mlp)
            t_mlp = self.dropout3(t_mlp)
            
            t_out = self.fc7(t_mlp)

        # =================== ENHANCED FUSION ===================
        # Prepare for attention-based fusion
        s_expanded = s_out.unsqueeze(1).expand(-1, self.pred_len, -1)  # [BC, pred_len, pred_len]
        t_expanded = t_out.unsqueeze(1).expand(-1, self.pred_len, -1)  # [BC, pred_len, pred_len]
        
        # Apply cross-attention between streams
        s_attended, _ = self.fusion_attention(s_expanded, t_expanded, t_expanded)
        t_attended, _ = self.fusion_attention(t_expanded, s_expanded, s_expanded)
        
        # Extract features
        s_fused = s_attended.mean(dim=1)  # [BC, pred_len]
        t_fused = t_attended.mean(dim=1)  # [BC, pred_len]
        
        # Learnable stream combination
        stream_weights = F.softmax(self.stream_weights, dim=0)
        
        # Concatenate and fuse
        combined = torch.cat((s_fused * stream_weights[0], t_fused * stream_weights[1]), dim=1)
        x = self.fc8(combined)
        x = self.final_highway(x)

        # =================== OUTPUT PROCESSING ===================
        # Channel concatenation with moving normalization
        x = torch.reshape(x, (B, C, self.pred_len))  # [Batch, Channel, Output]
        x = self.moving_norm(x)
        x = x.permute(0, 2, 1)  # to [Batch, Output, Channel]

        return x

class Network(nn.Module):
    """
    Original Network - kept for backward compatibility
    Enhanced with minimal changes
    """
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len

        # Non-linear Stream - Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len)//stride + 1
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            self.patch_num += 1

        # Enhanced components
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)
        self.dropout1 = nn.Dropout(0.1)  # Added dropout
        
        # CNN Depthwise with residual improvement
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Enhanced residual stream
        self.fc2 = nn.Linear(self.dim, patch_len)
        self.residual_scale = nn.Parameter(torch.ones(1))  # Learnable residual scaling

        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Enhanced flatten head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)
        self.dropout2 = nn.Dropout(0.1)  # Added dropout

        # Enhanced linear stream
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # Enhanced fusion
        self.fc8 = nn.Linear(pred_len * 2, pred_len)
        self.stream_balance = nn.Parameter(torch.tensor(0.5))  # Learnable stream balance

    def forward(self, s, t):
        # x: [Batch, Input, Channel]
        # s - seasonality, t - trend
        
        s = s.permute(0,2,1)  # to [Batch, Channel, Input]
        t = t.permute(0,2,1)  # to [Batch, Channel, Input]
        
        # Channel split for channel independence
        B = s.shape[0]  # Batch size
        C = s.shape[1]  # Channel size  
        I = s.shape[2]  # Input size
        s = torch.reshape(s, (B*C, I))  # [Batch and Channel, Input]
        t = torch.reshape(t, (B*C, I))  # [Batch and Channel, Input]

        # Non-linear Stream - Patching
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: [Batch and Channel, Patch_num, Patch_len]
        
        # Enhanced patch embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)
        s = self.dropout1(s)

        res = s

        # CNN Depthwise
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        # Enhanced residual stream with learnable scaling
        res = self.fc2(res)
        s = s + res * self.residual_scale

        # CNN Pointwise
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # Enhanced flatten head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.dropout2(s)
        s = self.fc4(s)

        # Linear Stream - Enhanced MLP
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)

        # Enhanced streams concatenation with learnable balance
        balance = torch.sigmoid(self.stream_balance)
        s_weighted = s * balance
        t_weighted = t * (1 - balance)
        
        x = torch.cat((s_weighted, t_weighted), dim=1)
        x = self.fc8(x)

        # Channel concatenation
        x = torch.reshape(x, (B, C, self.pred_len))  # [Batch, Channel, Output]
        x = x.permute(0,2,1)  # to [Batch, Output, Channel]

        return x