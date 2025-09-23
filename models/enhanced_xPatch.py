import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers.enhanced_decomp import EnhancedDECOMP
from layers.enhanced_transformer import EnhancedTransformerNetwork
from layers.enhanced_network import EnhancedNetwork
from layers.revin import RevIN
from layers.enhanced_modules import (
    AdaptiveInstanceNorm, CrossChannelFusion, CBAM, 
    LowRankLinear, HighwayConnection
)

class EnhancedModel(nn.Module):
    """
    Enhanced xPatch Model with all improvements:
    - Anti-distribution shift mechanisms
    - Enhanced inter-channel dependencies
    - Improved long-range modeling
    - Adaptive non-linearity
    - Scalable parameter sharing
    """
    def __init__(self, configs):
        super(EnhancedModel, self).__init__()

        # Parameters
        seq_len = configs.seq_len   
        pred_len = configs.pred_len 
        c_in = configs.enc_in       
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_in = c_in

        # Enhanced Transformer parameters
        d_model = getattr(configs, 'd_model', 512)
        nhead = getattr(configs, 'nhead', 8)
        num_layers = getattr(configs, 'num_layers', 3)
        dropout = getattr(configs, 'dropout', 0.1)

        # Patching parameters
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        # Enhanced normalization strategy
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)
        
        # Additional adaptive normalization layers
        self.adaptive_norm_input = AdaptiveInstanceNorm(c_in)
        self.adaptive_norm_output = AdaptiveInstanceNorm(c_in)

        # Moving Average parameters
        self.ma_type = configs.ma_type
        alpha = configs.alpha       
        beta = configs.beta         

        mb_k_small = getattr(configs, 'mb_k_small', 7)
        mb_k_large = getattr(configs, 'mb_k_large', 31)
        emd_imfs = getattr(configs, 'emd_imfs', 2)

        # Enhanced decomposition
        self.use_enhanced_decomp = getattr(configs, 'use_enhanced_decomp', True)
        if self.use_enhanced_decomp:
            self.decomp = EnhancedDECOMP(
                self.ma_type, alpha, beta,
                seq_len=seq_len, enc_in=c_in,
                mb_k_small=mb_k_small, mb_k_large=mb_k_large, emd_imfs=emd_imfs
            )
        else:
            from layers.decomp import DECOMP
            self.decomp = DECOMP(
                self.ma_type, alpha, beta,
                seq_len=seq_len, enc_in=c_in,
                mb_k_small=mb_k_small, mb_k_large=mb_k_large, emd_imfs=emd_imfs
            )

        # Enhanced network architecture
        self.use_enhanced_network = getattr(configs, 'use_enhanced_network', True)
        if self.use_enhanced_network:
            if c_in > 32:  # Use transformer for high-dimensional data
                self.net = EnhancedTransformerNetwork(
                    seq_len, pred_len, patch_len, stride, padding_patch, 
                    d_model, nhead, num_layers, dropout, c_in
                )
            else:  # Use enhanced CNN-based network for lower dimensions
                self.net = EnhancedNetwork(
                    seq_len, pred_len, patch_len, stride, padding_patch, c_in
                )
        else:
            from layers.transformer import TransformerNetwork
            self.net = TransformerNetwork(
                seq_len, pred_len, patch_len, stride, padding_patch, 
                d_model, nhead, num_layers, dropout
            )

        # Cross-channel processing for inter-variable dependencies
        if c_in > 1:
            self.cbam = CBAM(c_in, reduction=max(c_in // 16, 1))
            self.cross_channel_fusion = CrossChannelFusion(c_in, seq_len)
            
            # Learnable channel mixing for better scalability
            self.channel_mixer = nn.Parameter(
                torch.eye(c_in) * 0.8 + torch.ones(c_in, c_in) * 0.2 / c_in
            )
        
        # Enhanced prediction head with highway connections
        self.prediction_head = nn.Sequential(
            LowRankLinear(pred_len, pred_len * 2, rank=pred_len // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            LowRankLinear(pred_len * 2, pred_len, rank=pred_len // 4),
        )
        self.prediction_highway = HighwayConnection(pred_len)
        
        # Temporal consistency regularization
        self.temporal_consistency_weight = nn.Parameter(torch.tensor(0.1))
        
        # Multi-scale prediction for better accuracy
        self.multi_scale_predictor = nn.ModuleList([
            nn.Linear(pred_len, pred_len) for _ in range(3)
        ])
        self.scale_weights = nn.Parameter(torch.ones(3))
        
        # Adaptive loss weighting for different components
        self.loss_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.3]))  # [main, consistency, multi_scale]
        
    def apply_channel_processing(self, x):
        """Apply cross-channel processing if multiple channels"""
        if self.c_in > 1 and hasattr(self, 'cbam'):
            # x: [B, L, C] -> [B, C, L]
            x_channel = x.permute(0, 2, 1)
            
            # Apply channel attention
            x_channel = self.cbam(x_channel)
            
            # Apply cross-channel fusion
            x_channel = self.cross_channel_fusion(x_channel)
            
            # Apply learnable channel mixing
            x_mixed = torch.matmul(self.channel_mixer, x_channel)
            x_channel = x_channel + 0.2 * x_mixed
            
            # Back to [B, L, C]
            x = x_channel.permute(0, 2, 1)
        
        return x
    
    def apply_multi_scale_prediction(self, x):
        """Apply multi-scale prediction for robustness"""
        # x: [B, L, C]
        predictions = []
        
        for i, predictor in enumerate(self.multi_scale_predictor):
            # Apply different temporal scales
            if i == 0:  # Full resolution
                pred = predictor(x)
            elif i == 1:  # Half resolution
                x_down = F.avg_pool1d(x.transpose(1, 2), kernel_size=2).transpose(1, 2)
                pred = predictor(x_down)
                pred = F.interpolate(pred.transpose(1, 2), size=x.size(1)).transpose(1, 2)
            else:  # Quarter resolution
                x_down = F.avg_pool1d(x.transpose(1, 2), kernel_size=4).transpose(1, 2)
                pred = predictor(x_down)
                pred = F.interpolate(pred.transpose(1, 2), size=x.size(1)).transpose(1, 2)
            
            predictions.append(pred)
        
        # Weighted combination
        scale_weights = F.softmax(self.scale_weights, dim=0)
        combined = sum(w * pred for w, pred in zip(scale_weights, predictions))
        
        return combined, predictions
    
    def compute_temporal_consistency_loss(self, prediction, target=None):
        """Compute temporal consistency regularization"""
        if prediction.size(1) < 2:
            return torch.tensor(0.0, device=prediction.device)
        
        # Temporal smoothness loss
        diff = prediction[:, 1:] - prediction[:, :-1]
        consistency_loss = torch.mean(diff ** 2)
        
        # Add target consistency if available
        if target is not None and target.size(1) >= 2:
            target_diff = target[:, 1:] - target[:, :-1]
            target_consistency = torch.mean((diff - target_diff) ** 2)
            consistency_loss = consistency_loss + target_consistency
        
        return consistency_loss * torch.sigmoid(self.temporal_consistency_weight)
    
    def forward(self, x, return_loss_components=False):
        """
        Enhanced forward pass with multiple improvements
        x: [Batch, Input, Channel]
        """
        original_x = x.clone()
        
        # ================ INPUT PROCESSING ================
        # Apply RevIN normalization if enabled
        if self.revin:
            x = self.revin_layer(x, 'norm')
        
        # Apply adaptive normalization for distribution shift
        x_norm = x.permute(0, 2, 1)  # [B, C, L]
        x_norm = self.adaptive_norm_input(x_norm)
        x = x_norm.permute(0, 2, 1)  # [B, L, C]
        
        # Apply cross-channel processing
        x = self.apply_channel_processing(x)
        
        # ================ DECOMPOSITION ================
        if self.ma_type == 'reg':   # No decomposition
            seasonal_init, trend_init = x, x
        else:
            seasonal_init, trend_init = self.decomp(x)
        
        # ================ PREDICTION ================
        # Main prediction through enhanced network
        main_prediction = self.net(seasonal_init, trend_init)
        
        # Apply enhanced prediction head
        enhanced_pred = self.prediction_head(main_prediction)
        enhanced_pred = self.prediction_highway(enhanced_pred)
        
        # Combine main and enhanced predictions
        final_prediction = 0.7 * main_prediction + 0.3 * enhanced_pred
        
        # Apply multi-scale prediction
        multi_scale_pred, scale_predictions = self.apply_multi_scale_prediction(final_prediction)
        
        # Final combination
        prediction = 0.8 * final_prediction + 0.2 * multi_scale_pred
        
        # ================ OUTPUT PROCESSING ================
        # Apply adaptive output normalization
        pred_norm = prediction.permute(0, 2, 1)  # [B, C, L]
        pred_norm = self.adaptive_norm_output(pred_norm)
        prediction = pred_norm.permute(0, 2, 1)  # [B, L, C]
        
        # Apply RevIN denormalization if enabled
        if self.revin:
            prediction = self.revin_layer(prediction, 'denorm')
        
        # ================ LOSS COMPONENTS ================
        if return_loss_components:
            # Compute additional loss components for training
            consistency_loss = self.compute_temporal_consistency_loss(prediction, original_x)
            
            # Multi-scale consistency loss
            multi_scale_losses = []
            for scale_pred in scale_predictions:
                if self.revin:
                    scale_pred_denorm = self.revin_layer(scale_pred, 'denorm')
                else:
                    scale_pred_denorm = scale_pred
                scale_loss = F.mse_loss(scale_pred_denorm, prediction.detach())
                multi_scale_losses.append(scale_loss)
            
            avg_multi_scale_loss = torch.stack(multi_scale_losses).mean()
            
            return prediction, {
                'consistency_loss': consistency_loss,
                'multi_scale_loss': avg_multi_scale_loss,
                'seasonal_component': seasonal_init,
                'trend_component': trend_init,
                'loss_weights': self.loss_weights
            }
        
        return prediction

class Model(nn.Module):
    """
    Main xPatch Model - Enhanced version with backward compatibility
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        # Check if enhanced mode is requested
        self.use_enhanced = getattr(configs, 'use_enhanced', False)
        
        if self.use_enhanced:
            self.model = EnhancedModel(configs)
        else:
            # Original implementation with minimal enhancements
            seq_len = configs.seq_len   
            pred_len = configs.pred_len 
            c_in = configs.enc_in       

            # Transformer parameters
            d_model = getattr(configs, 'd_model', 512)
            nhead = getattr(configs, 'nhead', 8)
            num_layers = getattr(configs, 'num_layers', 3)
            dropout = getattr(configs, 'dropout', 0.1)

            # Patching
            patch_len = configs.patch_len
            stride = configs.stride
            padding_patch = configs.padding_patch

            # Normalization
            self.revin = configs.revin
            self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)

            # Moving Average
            self.ma_type = configs.ma_type
            alpha = configs.alpha       
            beta = configs.beta         

            mb_k_small = getattr(configs, 'mb_k_small', 7)
            mb_k_large = getattr(configs, 'mb_k_large', 31)
            emd_imfs   = getattr(configs, 'emd_imfs', 2)

            from layers.decomp import DECOMP
            from layers.transformer import TransformerNetwork
            
            self.decomp = DECOMP(self.ma_type, alpha, beta,
                               seq_len=seq_len, enc_in=c_in,
                               mb_k_small=mb_k_small, mb_k_large=mb_k_large, emd_imfs=emd_imfs)
            self.net = TransformerNetwork(seq_len, pred_len, patch_len, stride, padding_patch, 
                                        d_model, nhead, num_layers, dropout)
            
            # Add minimal enhancements
            self.dropout = nn.Dropout(dropout)
            self.output_projection = nn.Linear(pred_len, pred_len)

    def forward(self, x):
        """Forward pass - delegates to enhanced or original model"""
        if self.use_enhanced:
            return self.model(x)
        else:
            # Original implementation with minimal enhancements
            if self.revin:
                x = self.revin_layer(x, 'norm')

            if self.ma_type == 'reg':   
                x = self.net(x, x)
            else:
                seasonal_init, trend_init = self.decomp(x)
                x = self.net(seasonal_init, trend_init)

            # Apply minimal enhancements
            x = self.dropout(x)
            x = x + self.output_projection(x)  # Residual connection

            if self.revin:
                x = self.revin_layer(x, 'denorm')

            return x