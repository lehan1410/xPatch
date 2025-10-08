import torch
from torch import nn

class channel_attn_block(nn.Module):
    def __init__(self, seq_len, d_model, dropout):
        super(channel_attn_block, self).__init__()
        self.channel_att_norm = nn.BatchNorm1d(seq_len)
        self.fft_norm = nn.LayerNorm(d_model)
        self.channel_attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.fft_layer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x):
        # x: [B, seq_len, d_model]
        attn_out, _ = self.channel_attn(x, x, x)
        res_2 = self.channel_att_norm(attn_out + x)
        res_2 = self.fft_norm(self.fft_layer(res_2) + res_2)
        return res_2

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, d_model, dropout=0.1, n_layers=2):
        super(Network, self).__init__()

        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in  = c_in
        self.d_model = d_model
        self.n_layers = n_layers

        self.channel_proj = nn.Linear(self.enc_in, self.d_model)
        self.channel_attn_blocks = nn.ModuleList([
            channel_attn_block(self.seq_len, self.d_model, dropout)
            for _ in range(self.n_layers)
        ])

        self.mlp = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.pred_len)
        )

        self.out_proj = nn.Linear(self.d_model, self.enc_in)

        # Linear Stream
        self.fc5 = nn.Linear(seq_len, pred_len * 2)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        B, I, C = s.shape
        s_proj = self.channel_proj(s)  # [B, Input, d_model]

        # Multi-layer Channel Attention
        for i in range(self.n_layers):
            s_proj = self.channel_attn_blocks[i](s_proj)  # [B, Input, d_model]

        # MLP dự báo
        y = self.mlp(s_proj.transpose(1,2))  # [B, d_model, pred_len]
        y = y.transpose(1,2)  # [B, pred_len, d_model]
        y = self.out_proj(y)  # [B, pred_len, C]

        # Linear Stream
        t = t.permute(0,2,1) # [B, C, Input]
        t = torch.reshape(t, (B*C, I))
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [B, pred_len, C]

        return t + y