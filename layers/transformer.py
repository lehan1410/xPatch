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
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, d_model=512, nhead=8, num_layers=3, dropout=0.1):
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
        
        self.t_patch_len = max(patch_len, patch_len * 2)
        self.t_stride = max(stride, stride * 2)
        self.t_padding_patch = padding_patch
        self.t_patch_num = (seq_len - self.t_patch_len) // self.t_stride + 1
        if self.t_padding_patch == 'end':
            self.t_padding_patch_layer = nn.ReplicationPad1d((0, self.t_stride))
            self.t_patch_num += 1
        else:
            self.t_padding_patch_layer = None
        
        # Input projection
        self.input_projection = nn.Linear(self.patch_len, d_model)  # Project single feature to d_model dimensions
        
        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                    dim_feedforward=4*d_model, dropout=dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.token_norm_s = nn.LayerNorm(d_model)
        self.dropout_s = nn.Dropout(dropout)

        # Decoder (prediction head)
        self.decoder = nn.Sequential(
            nn.Linear(d_model * self.patch_num, pred_len * 2),
            nn.GELU(),
            nn.Linear(pred_len * 2, pred_len)
        )

        t_layers = max(1, num_layers // 2)
        self.t_input_projection = nn.Linear(self.t_patch_len, d_model)
        t_enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                 dim_feedforward=4*d_model, dropout=dropout, batch_first=False)
        self.t_transformer_encoder = nn.TransformerEncoder(t_enc_layer, num_layers=t_layers)
        self.token_norm_t = nn.LayerNorm(d_model)
        self.dropout_t = nn.Dropout(dropout)
        self.t_decoder = nn.Sequential(
            nn.Linear(d_model * self.t_patch_num, pred_len * 2),
            nn.GELU(),
            nn.Linear(pred_len * 2, pred_len)
        )

        # Cross-fusion: season attends to trend tokens
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.cross_ln = nn.LayerNorm(d_model)
        self.cross_drop = nn.Dropout(dropout)


        # self.fc5 = nn.Linear(seq_len, pred_len * 4)
        # self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        # self.ln1 = nn.LayerNorm(pred_len * 2)
        # self.fc6 = nn.Linear(pred_len * 2, pred_len)
        # self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        # self.ln2 = nn.LayerNorm(pred_len // 2)
        # self.fc7 = nn.Linear(pred_len // 2, pred_len)
        
        # --- Final concat layer ---
        self.fc8 = nn.Linear(pred_len * 2, pred_len)
        self.gate_fc = nn.Linear(2, 1)
    
    def _patch(self, x):
        # x: [B, C, seq_len] -> [B*C, patch_num, patch_len]
        B, C, I = x.shape
        x = x.reshape(B * C, I)
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return x, B, C
    
    def _patch_trend(self, x):
        # x: [B, C, seq_len] -> [B*C, t_patch_num, t_patch_len]
        B, C, I = x.shape
        x = x.reshape(B * C, I)
        if self.t_padding_patch == 'end':
            x = self.t_padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.t_patch_len, step=self.t_stride)
        return x, B, C
    
    def _encode_season_tokens(self, s):
        # s: [B, I, C] -> tokens [P, B*C, d]
        B, I, C = s.shape
        s = s.permute(0, 2, 1)                     # [B, C, I]
        s, _, _ = self._patch(s)                    # [B*C, P, Lp]
        s = self.input_projection(s)               # [B*C, P, d]
        s = self.pos_encoder(s)
        s = self.token_norm_s(s)
        s = self.dropout_s(s)
        s = s.permute(1, 0, 2)                     # [P, B*C, d]
        s = self.transformer_encoder(s)            # [P, B*C, d]
        return s
    
    def _encode_trend_tokens(self, t):
        # t: [B, I, C] -> tokens [Pt, B*C, d]
        B, I, C = t.shape
        t = t.permute(0, 2, 1)                     # [B, C, I]
        t, _, _ = self._patch_trend(t)              # [B*C, Pt, Lt]
        t = self.t_input_projection(t)             # [B*C, Pt, d]
        t = self.pos_encoder(t)                    # reuse PE
        t = self.token_norm_t(t)
        t = self.dropout_t(t)
        t = t.permute(1, 0, 2)                     # [Pt, B*C, d]
        t = self.t_transformer_encoder(t)
        return t
    
    def _decode(self, tokens, is_trend=False, B=None, C=None):
        # tokens: [P, B*C, d] -> [B, pred_len, C]
        P, BC, d = tokens.shape
        x = tokens.permute(1, 0, 2).reshape(BC, P * d)  # [B*C, P*d]
        if is_trend:
            x = self.t_decoder(x)
        else:
            x = self.decoder(x)
        x = x.view(B, C, self.pred_len).permute(0, 2, 1)
        return x

    def forward(self, s, t):
        # s, t: [Batch, Input, Channel]
        B, I, C = s.shape

        s_tok = self._encode_season_tokens(s)        # [P,  B*C, d]
        t_tok = self._encode_trend_tokens(t)

        s_q = s_tok
        t_kv = t_tok
        s_fused, _ = self.cross_attn(query=s_q, key=t_kv, value=t_kv, need_weights=False)
        s_tok = self.cross_ln(s_tok + self.cross_drop(s_fused))  # residual

        # Decode to predictions
        s_out = self._decode(s_tok, is_trend=False, B=B, C=C)    # [B, pred_len, C]
        t_out = self._decode(t_tok, is_trend=True,  B=B, C=C)    # [B, pred_len, C]

        # Gate fusion per-time per-channel
        g_in = torch.stack([s_out, t_out], dim=-1)               # [B, pred_len, C, 2]
        g = torch.sigmoid(self.gate_fc(g_in)).squeeze(-1)        # [B, pred_len, C]
        s_w = g * s_out
        t_w = (1.0 - g) * t_out

        # Projection head
        x = torch.cat([s_w, t_w], dim=1)                         # [B, 2*pred_len, C]
        x = x.permute(0, 2, 1)                                   # [B, C, 2*pred_len]
        x = self.fc8(x).permute(0, 2, 1)                         # [B, pred_len, C]
        return x
        
        # # Process each channel independently
        # outputs = []
        # for i in range(C):
        #     # Extract single channel and reshape
        #     x = s[:, :, i].unsqueeze(-1)  # [Batch, Input, 1]
            
        #     # Project input to d_model dimensions
        #     x = self.input_projection(x)  # [Batch, Input, d_model]
            
        #     # Add positional encoding
        #     x = self.pos_encoder(x)
            
        #     # Transformer expects: [Input, Batch, d_model]
        #     x = x.permute(1, 0, 2)
            
        #     # Pass through transformer
        #     x = self.transformer_encoder(x)
            
        #     # Reshape back: [Batch, Input, d_model]
        #     x = x.permute(1, 0, 2)
            
        #     # Flatten and decode
        #     x = x.reshape(B, -1)  # [Batch, Input * d_model]
        #     x = self.decoder(x)  # [Batch, pred_len]
            
        #     outputs.append(x)
        
        # # Stack all channel outputs
        # x = torch.stack(outputs, dim=-1)  # [Batch, pred_len, Channel]
        
        # return x 

        # # --- PATCHING SEASONALITY ---
        # s = s.permute(0, 2, 1)  # [B, C, seq_len]
        # s = s.reshape(B * C, I) # [B*C, seq_len]
        # if self.padding_patch == 'end':
        #     s = self.padding_patch_layer(s)
        # s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B*C, patch_num, patch_len]

        # # Patch Embedding: apply input_projection for each patch
        # s = self.input_projection(s)  # [B*C, patch_num, patch_len, d_model]

        # # Positional Encoding
        # s = self.pos_encoder(s)  # [B*C, patch_num, d_model]

        # # Transformer expects: [patch_num, B*C, d_model]
        # s = s.permute(1, 0, 2)  # [patch_num, B*C, d_model]
        # s = self.transformer_encoder(s)
        # s = s.permute(1, 0, 2)  # [B*C, patch_num, d_model]

        # # Flatten and decode
        # s = s.reshape(B * C, -1)  # [B*C, patch_num*d_model]
        # # Adjust decoder input size if needed
        # s = self.decoder(s)  # [B*C, pred_len]
        # s = s.view(B, C, self.pred_len).permute(0, 2, 1)  # [B, pred_len, C]

        # # # Season
        # # s = s.permute(0, 2, 1).reshape(B * C, I, 1)  # -> [B*C, seq_len, 1]
        # # s = self.input_projection(s)                # -> [B*C, seq_len, d_model]
        # # s = self.pos_encoder(s)
        # # s = s.permute(1, 0, 2)                      # -> [seq_len, B*C, d_model]
        # # s = self.transformer_encoder(s)
        # # s = s.permute(1, 0, 2)                      # -> [B*C, seq_len, d_model]
        # # s = s.reshape(B * C, -1)                    # -> [B*C, seq_len*d_model]
        # # s = self.decoder(s)                         # -> [B*C, pred_len]
        # # s = s.view(B, C, self.pred_len).permute(0, 2, 1)  # -> [B, pred_len, C]

        # # =============== Trend (MLP) ============================
        # t = t.permute(0, 2, 1).reshape(B * C, I)     # -> [B*C, seq_len]
        # t = self.fc5(t)                              # -> [B*C, pred_len * 4]
        # t = self.avgpool1(t.unsqueeze(1)).squeeze(1) # -> [B*C, pred_len * 2]
        # t = self.ln1(t)
        # t = self.fc6(t)
        # t = self.avgpool2(t.unsqueeze(1)).squeeze(1) # -> [B*C, pred_len // 2]
        # t = self.ln2(t)
        # t = self.fc7(t)                              # -> [B*C, pred_len]
        # t = t.view(B, C, self.pred_len).permute(0, 2, 1)  # -> [B, pred_len, C]

        # g_in = torch.stack([s, t], dim=-1)          # [B, pred_len, C, 2]
        # g = torch.sigmoid(self.gate_fc(g_in)).squeeze(-1)  # [B, pred_len, C]
        # s_w = g * s
        # t_w = (1.0 - g) * t

        # # =============== Fusion =============================
        # x = torch.cat([s, t], dim=1)                 # [B, pred_len*2, C]
        # x = x.permute(0, 2, 1)                       # [B, C, pred_len*2]
        # x = self.fc8(x)                              # [B, C, pred_len]
        # x = x.permute(0, 2, 1)                       # [B, pred_len, C]

        # return x