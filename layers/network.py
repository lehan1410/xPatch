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

class ar_self_attn_block(nn.Module):
    def __init__(self, enc_in, seq_len, d_model, dropout):
        super(ar_self_attn_block, self).__init__()
        self.input_proj = nn.Linear(enc_in, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=1, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, enc_in)  # Linear để điều chỉnh về số channel

    def forward(self, x):
        # x: [B, seq_len, enc_in]
        x_embed = self.input_proj(x)  # [B, seq_len, d_model]
        seq_len = x_embed.size(1)
        attn_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        attn_out, _ = self.self_attn(x_embed, x_embed, x_embed, attn_mask=attn_mask)
        x = self.norm1(attn_out + x_embed)
        x = self.norm2(self.ffn(x) + x)
        x = self.out_proj(x)  # [B, seq_len, enc_in]
        x = x.permute(0, 2, 1)  # [B, enc_in, seq_len]
        return x

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

        self.ar_attn_blocks = nn.ModuleList([
            ar_self_attn_block(self.enc_in, self.seq_len, self.d_model, dropout)
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

        # Conv branch (depthwise)
        s_conv = s.permute(0, 2, 1)  # [B, Channel, Input]
        s_conv = self.conv1d(s_conv)  # [B, C, seq_len]
        s_pool = self.pool(s_conv)  # [B, C, seq_len]

        # Auto-Regressive Self-Attention branch
        s_ar = s  # [B, Input, Channel]
        s_ar = s_ar.permute(0, 1, 2)
        for i in range(self.n_layers):
            s_ar = self.ar_attn_blocks[i](s_ar)  # [B, enc_in, seq_len]

        # Tổng hợp đặc trưng attention, conv, AR self-attn
        fused_seq = attn_seq + s_pool + s_conv + s_ar # [B, enc_in, seq_len]

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