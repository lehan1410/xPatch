import torch
from torch import nn

class channel_attn_block(nn.Module):
    def __init__(self, enc_in, d_model, dropout):
        super(channel_attn_block, self).__init__()
        self.channel_att_norm = nn.BatchNorm1d(enc_in)
        self.fft_norm = nn.LayerNorm(d_model)
        self.channel_attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
    
        self.fft_layer = nn.Sequential(
            nn.Linear(d_model, int(d_model*2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model*2), d_model),
        )
    def forward(self, residual):
        attn_out, _ = self.channel_attn(residual, residual, residual)  # [B, Channel, d_model]
        attn_out = attn_out + residual
        res_2 = self.channel_att_norm(attn_out)
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
        self.period_len = period_len

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.channel_proj = nn.Linear(self.seq_len, self.d_model)
        self.channel_attn_blocks = nn.ModuleList([
            channel_attn_block(self.enc_in, self.d_model, dropout)
            for _ in range(self.n_layers)
        ])

        self.conv1d = nn.Conv1d(
            in_channels=self.enc_in, out_channels=self.enc_in,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False, groups=self.enc_in
        )

        self.pool = nn.AvgPool1d(
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2
        )

        self.norm_attn = nn.LayerNorm(self.seq_len)
        self.norm_conv = nn.LayerNorm(self.seq_len)
        self.norm_pool = nn.LayerNorm(self.seq_len)
        self.norm_raw  = nn.LayerNorm(self.seq_len)

        # Chuyển từ d_model về seq_len để dùng MLP như yêu cầu
        self.to_seq = nn.Linear(self.d_model, self.seq_len)

        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.seg_num_y)
        )

        self.out_proj = nn.Linear(self.pred_len, self.enc_in)

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

        # Attention branch
        s_proj = s.permute(0, 2, 1)  # [B, Channel, Input]
        s_proj = self.channel_proj(s_proj)  # [B, Channel, d_model]
        for i in range(self.n_layers):
            s_proj = self.channel_attn_blocks[i](s_proj)  # [B, Channel, d_model]
        attn_seq = self.to_seq(s_proj)  # [B, Channel, seq_len]
        attn_seq = self.norm_attn(attn_seq)

        # Conv branch (depthwise)
        s = s.permute(0, 2, 1)  # [B, Channel, Input]
        s_conv = self.conv1d(s)  # [B, C, seq_len]
        s_conv = self.norm_conv(s_conv)
        s_pool = self.pool(s)  # [B, C, seq_len]
        s_pool = self.norm_pool(s_pool)
        s_norm = self.norm_raw(s)

        # Tổng hợp đặc trưng attention và conv
        fused_seq = attn_seq + s_pool + s_conv + s_norm  # [B, Channel, seq_len]

        # Reshape để dùng MLP như yêu cầu
        fused_seq = fused_seq.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)  # [B*C, period_len, seg_num_x]
        y = self.mlp(fused_seq)  # [B*C, period_len, seg_num_y]
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)  # [B, Channel, pred_len]
        y = y.permute(0, 2, 1)  # [B, pred_len, Channel]

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