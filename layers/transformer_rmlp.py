import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from layers.Invertible import RevIN

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerRMLPNetwork(nn.Module):
    """
    Network combining Transformer for seasonal component and RMLP for trend component
    """
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, d_model=512, nhead=8, num_layers=3, dropout=0.1, configs=None):
        super(TransformerRMLPNetwork, self).__init__()
        
        # Parameters
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        # Patching params for seasonal component
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1
        else:
            self.padding_patch_layer = None
        
        # Transformer components for seasonal
        self.input_projection = nn.Linear(self.patch_len, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                  dim_feedforward=4*d_model, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Decoder for seasonal
        self.decoder = nn.Sequential(
            nn.Linear(d_model * self.patch_num, pred_len * 2),
            nn.GELU(),
            nn.Linear(pred_len * 2, pred_len)
        )

        # RMLP components for trend
        self.temporal = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, seq_len)
        )
        self.projection = nn.Linear(seq_len, pred_len)
        
        # RevIN for RMLP
        if configs and hasattr(configs, 'channel'):
            self.channel = configs.channel
            self.rev = RevIN(configs.channel) if getattr(configs, 'rev', True) else None
        else:
            self.channel = None  
            self.rev = None
        
        # Final fusion layer
        self.fc_fusion = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s, t):
        # s: seasonal component, t: trend component
        # Both: [Batch, Input, Channel]
        B, I, C = s.shape
        
        # Initialize RevIN if needed (for trend processing)
        if self.rev is None and (self.channel is None or self.channel != C):
            self.channel = C
            self.rev = RevIN(C)
        
        # =============== Seasonal (Transformer) ============================
        s = s.permute(0, 2, 1)  # [B, C, seq_len]
        s = s.reshape(B * C, I) # [B*C, seq_len]
        
        # Patching
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B*C, patch_num, patch_len]

        # Patch Embedding
        s = self.input_projection(s)  # [B*C, patch_num, d_model]

        # Positional Encoding
        s = self.pos_encoder(s)  # [B*C, patch_num, d_model]

        # Transformer processing
        s = s.permute(1, 0, 2)  # [patch_num, B*C, d_model]
        s = self.transformer_encoder(s)
        s = s.permute(1, 0, 2)  # [B*C, patch_num, d_model]

        # Flatten and decode
        s = s.reshape(B * C, -1)  # [B*C, patch_num*d_model]
        s = self.decoder(s)  # [B*C, pred_len]
        s = s.view(B, C, self.pred_len).permute(0, 2, 1)  # [B, pred_len, C]

        # =============== Trend (RMLP) ============================
        # Apply RevIN normalization
        t_norm = self.rev(t, 'norm') if self.rev else t  # [B, I, C]
        
        # Apply temporal transformation with residual
        t_res = t_norm.transpose(1, 2)  # [B, C, I]
        t_res = self.temporal(t_res)    # [B, C, I]
        t_res = t_res.transpose(1, 2)    # [B, I, C]
        
        # Add residual connection
        t_norm = t_norm + t_res
        
        # Project to prediction length
        t_proj = t_norm.transpose(1, 2)  # [B, C, I]
        t_proj = self.projection(t_proj)  # [B, C, pred_len]
        t_proj = t_proj.transpose(1, 2)   # [B, pred_len, C]
        
        # Apply RevIN denormalization
        t = self.rev(t_proj, 'denorm') if self.rev else t_proj  # [B, pred_len, C]

        # =============== Fusion =============================
        x = torch.cat([s, t], dim=1)  # [B, pred_len*2, C]
        x = x.permute(0, 2, 1)        # [B, C, pred_len*2]
        x = self.fc_fusion(x)         # [B, C, pred_len]
        x = x.permute(0, 2, 1)        # [B, pred_len, C]

        return x 