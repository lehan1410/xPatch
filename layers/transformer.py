import torch
import torch.nn as nn
import math

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

class TransformerNetwork(nn.Module):
    def __init__(self, seq_len, pred_len, d_model=512, nhead=8, num_layers=3, dropout=0.1):
        super(TransformerNetwork, self).__init__()
        
        # Parameters
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(1, d_model)  # Project single feature to d_model dimensions
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                  dim_feedforward=4*d_model, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Decoder (prediction head)
        self.decoder = nn.Sequential(
            nn.Linear(d_model * seq_len, pred_len * 2),
            nn.GELU(),
            nn.Linear(pred_len * 2, pred_len)
        )

    def forward(self, s, t):
        # s, t: [Batch, Input, Channel]
        B = s.shape[0]  # Batch size
        C = s.shape[2]  # Channel size
        
        # Process each channel independently
        outputs = []
        for i in range(C):
            # Extract single channel and reshape
            x = s[:, :, i].unsqueeze(-1)  # [Batch, Input, 1]
            
            # Project input to d_model dimensions
            x = self.input_projection(x)  # [Batch, Input, d_model]
            
            # Add positional encoding
            x = self.pos_encoder(x)
            
            # Transformer expects: [Input, Batch, d_model]
            x = x.permute(1, 0, 2)
            
            # Pass through transformer
            x = self.transformer_encoder(x)
            
            # Reshape back: [Batch, Input, d_model]
            x = x.permute(1, 0, 2)
            
            # Flatten and decode
            x = x.reshape(B, -1)  # [Batch, Input * d_model]
            x = self.decoder(x)  # [Batch, pred_len]
            
            outputs.append(x)
        
        # Stack all channel outputs
        x = torch.stack(outputs, dim=-1)  # [Batch, pred_len, Channel]
        
        return x 