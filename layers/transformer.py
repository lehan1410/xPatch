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
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B*, P, d_model]
        return x + self.pe[:, :x.size(1)]

class TransformerNetwork(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 d_model=512, nhead=8, num_layers=3, dropout=0.1,
                 token_stride_s: int = 2, token_stride_t: int = 2):
        super(TransformerNetwork, self).__init__()
        
        # Params
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        # Season patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1
        else:
            self.padding_patch_layer = None

        # Trend patching (rộng hơn, ít token hơn)
        self.t_patch_len = max(patch_len, patch_len * 2)
        self.t_stride = max(stride, stride * 2)
        self.t_padding_patch = padding_patch
        self.t_patch_num = (seq_len - self.t_patch_len) // self.t_stride + 1
        if self.t_padding_patch == 'end':
            self.t_padding_patch_layer = nn.ReplicationPad1d((0, self.t_stride))
            self.t_patch_num += 1
        else:
            self.t_padding_patch_layer = None

        # Token downsample để tăng tốc
        self.token_stride_s = max(1, int(token_stride_s))
        self.token_stride_t = max(1, int(token_stride_t))
        self.patch_num_eff   = (self.patch_num + self.token_stride_s - 1) // self.token_stride_s
        self.t_patch_num_eff = (self.t_patch_num + self.token_stride_t - 1) // self.token_stride_t

        # Season branch
        self.input_projection = nn.Linear(self.patch_len, d_model, bias=True)
        self.pos_encoder = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.token_norm_s = nn.LayerNorm(d_model)
        self.dropout_s = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(d_model * self.patch_num_eff, pred_len * 2),
            nn.GELU(),
            nn.Linear(pred_len * 2, pred_len)
        )

        # Trend branch (nhẹ hơn)
        t_layers = max(1, num_layers // 2)
        self.t_input_projection = nn.Linear(self.t_patch_len, d_model, bias=True)
        t_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True
        )
        self.t_transformer_encoder = nn.TransformerEncoder(t_enc_layer, num_layers=t_layers)
        self.token_norm_t = nn.LayerNorm(d_model)
        self.dropout_t = nn.Dropout(dropout)
        self.t_decoder = nn.Sequential(
            nn.Linear(d_model * self.t_patch_num_eff, pred_len * 2),
            nn.GELU(),
            nn.Linear(pred_len * 2, pred_len)
        )

        # Cross-fusion: season attends to trend tokens
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_ln = nn.LayerNorm(d_model)
        self.cross_drop = nn.Dropout(dropout)

        # Fusion head
        self.fc8 = nn.Linear(pred_len * 2, pred_len)
        self.gate_fc = nn.Linear(2, 1)

    def _patch(self, x):
        # x: [B, C, seq_len] -> [B*C, P, Lp]
        B, C, I = x.shape
        x = x.reshape(B * C, I)
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return x, B, C

    def _patch_trend(self, x):
        # x: [B, C, seq_len] -> [B*C, Pt, Lt]
        B, C, I = x.shape
        x = x.reshape(B * C, I)
        if self.t_padding_patch == 'end':
            x = self.t_padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.t_patch_len, step=self.t_stride)
        return x, B, C

    def _encode_season_tokens(self, s):
        # s: [B, I, C] -> tokens [B*C, P_eff, d]
        B, I, C = s.shape
        s = s.permute(0, 2, 1)             # [B, C, I]
        s, _, _ = self._patch(s)           # [B*C, P, Lp]
        s = self.input_projection(s)       # [B*C, P, d]
        if self.token_stride_s > 1:
            s = s[:, ::self.token_stride_s, :]
        s = self.pos_encoder(s)
        s = self.token_norm_s(s)
        s = self.dropout_s(s)
        s = self.transformer_encoder(s)    # [B*C, P_eff, d]
        return s

    def _encode_trend_tokens(self, t):
        # t: [B, I, C] -> tokens [B*C, Pt_eff, d]
        B, I, C = t.shape
        t = t.permute(0, 2, 1)             # [B, C, I]
        t, _, _ = self._patch_trend(t)     # [B*C, Pt, Lt]
        t = self.t_input_projection(t)     # [B*C, Pt, d]
        if self.token_stride_t > 1:
            t = t[:, ::self.token_stride_t, :]
        t = self.pos_encoder(t)
        t = self.token_norm_t(t)
        t = self.dropout_t(t)
        t = self.t_transformer_encoder(t)  # [B*C, Pt_eff, d]
        return t

    def _decode(self, tokens, is_trend=False, B=None, C=None):
        # tokens: [B*C, P_eff, d] -> [B, pred_len, C]
        BC, P, d = tokens.shape
        x = tokens.reshape(BC, P * d)
        x = self.t_decoder(x) if is_trend else self.decoder(x)
        x = x.view(B, C, self.pred_len).permute(0, 2, 1)
        return x

    def forward(self, s, t):
        # s, t: [B, Input, C]
        B, I, C = s.shape

        # Encode
        s_tok = self._encode_season_tokens(s)    # [B*C, P_eff, d]
        t_tok = self._encode_trend_tokens(t)     # [B*C, Pt_eff, d]

        # Cross-attention: season attends to trend
        s_fused, _ = self.cross_attn(query=s_tok, key=t_tok, value=t_tok, need_weights=False)
        s_tok = self.cross_ln(s_tok + self.cross_drop(s_fused))

        # Decode
        s_out = self._decode(s_tok, is_trend=False, B=B, C=C)  # [B, pred_len, C]
        t_out = self._decode(t_tok, is_trend=True,  B=B, C=C)  # [B, pred_len, C]

        # Gate per-time per-channel
        g_in = torch.stack([s_out, t_out], dim=-1)  # [B, pred_len, C, 2]
        g = torch.sigmoid(self.gate_fc(g_in)).squeeze(-1)
        s_w = g * s_out
        t_w = (1.0 - g) * t_out

        # Projection head
        x = torch.cat([s_w, t_w], dim=1)           # [B, 2*pred_len, C]
        x = x.permute(0, 2, 1)                     # [B, C, 2*pred_len]
        x = self.fc8(x).permute(0, 2, 1)           # [B, pred_len, C]
        return x