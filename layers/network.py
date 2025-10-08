import torch
from torch import nn

class channel_attn_block(nn.Module):
    def __init__(self, enc_in, d_model, dropout):
        super(channel_attn_block, self).__init__()
        self.channel_att_norm = nn.BatchNorm1d(enc_in)
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
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model, dropout=0.1):
        super(Network, self).__init__()

        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in  = c_in
        self.period_len = period_len
        self.d_model = d_model

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(
            in_channels=self.enc_in, out_channels=self.d_model,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False, groups=1
        )

        self.pool = nn.AvgPool1d(
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2
        )

        self.channel_attn = channel_attn_block(self.seq_len, self.d_model, dropout)

        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.seg_num_y)
        )

        # Chuyển từ d_model về enc_in (C)
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
        s = s.permute(0,2,1) # [Batch, Channel, Input]
        t = t.permute(0,2,1) # [Batch, Channel, Input]

        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        t = torch.reshape(t, (B*C, I))

        # Seasonal Stream: Conv1d + Pooling + Channel Attention
        s_conv = self.conv1d(s)  # [B, d_model, seq_len]
        s_pool = self.pool(s_conv)  # [B, d_model, seq_len]
        s_pool = s_pool.permute(0,2,1)  # [B, seq_len, d_model]
        s_attn = self.channel_attn(s_pool)  # [B, seq_len, d_model]
        s = s_attn.permute(0,2,1)  # [B, d_model, seq_len]

        s = s.reshape(B*self.d_model, self.seg_num_x, self.period_len).permute(0, 2, 1)
        y = self.mlp(s)
        y = y.permute(0, 2, 1).reshape(B, self.d_model, self.pred_len)
        y = y.permute(0, 2, 1)  # [B, pred_len, d_model]

        # Chuyển về đúng số channel [B, pred_len, C]
        y = self.out_proj(y)  # [B, pred_len, C]

        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [Batch, pred_len, Channel] = [B, pred_len, C]

        return t + y