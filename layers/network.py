import torch
from torch import nn

class channel_attn_block(nn.Module):
    def __init__(self, enc_in, d_model, dropout):
        super(channel_attn_block, self).__init__()
        self.channel_att_norm = nn.BatchNorm1d(enc_in)
        self.fft_norm = nn.LayerNorm(d_model)
        # Attention trên channel: mỗi channel là một token, embedding là d_model
        self.channel_attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.fft_layer = nn.Sequential(
            nn.Linear(d_model, int(d_model*2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model*2), d_model),
        )
    def forward(self, residual):
        # residual: [B, Channel, d_model]
        # Attention trên channel: mỗi channel là một token
        # MultiheadAttention yêu cầu [B, Channel, d_model] với batch_first=True
        attn_out, _ = self.channel_attn(residual, residual, residual)  # [B, Channel, d_model]
        # BatchNorm1d expects [B, enc_in, d_model], normalize trên channel
        res_2 = self.channel_att_norm(attn_out)  # [B, Channel, d_model]
        res_2 = self.fft_norm(self.fft_layer(res_2) + res_2)
        return res_2

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model, dropout=0.1, n_layers=2):
        super(Network, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in  = c_in
        self.d_model = d_model
        self.n_layers = n_layers

        self.channel_proj = nn.Linear(self.seq_len, self.d_model)
        self.channel_attn_blocks = nn.ModuleList([
            channel_attn_block(self.enc_in, self.d_model, dropout)
            for _ in range(self.n_layers)
        ])

        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.pred_len)
        )

        self.out_proj = nn.Linear(self.pred_len, self.enc_in)

        # Linear Stream
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        B, I, C = s.shape
        # Đổi sang [B, Channel, Input] để attention trên channel
        s_proj = s.permute(0, 2, 1)  # [B, Channel, Input]
        s_proj = self.channel_proj(s_proj)  # [B, Channel, d_model]

        # Multi-layer Channel Attention (trên channel)
        for i in range(self.n_layers):
            s_proj = self.channel_attn_blocks[i](s_proj)  # [B, Channel, d_model]

        # MLP dự báo
        y = self.mlp(s_proj)  # [B, Channel, pred_len]
        y = y.permute(0, 2, 1)  # [B, pred_len, Channel]
        # Nếu muốn đầu ra [B, pred_len, Channel], không cần out_proj nữa
        # Nếu muốn [B, pred_len, enc_in], dùng out_proj
        # y = self.out_proj(y)    # [B, pred_len, enc_in] (nếu cần)

        # Linear Stream
        t = t.permute(0,2,1) # [B, C, Input]
        t = torch.reshape(t, (B*C, I))
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [B, pred_len, C]

        return t + y