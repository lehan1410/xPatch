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

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, residual):
        # residual: [B, Channel, d_model]
        # Attention trên channel: mỗi channel là một token
        # MultiheadAttention yêu cầu [B, Channel, d_model] với batch_first=True
        attn_out, _ = self.channel_attn(residual, residual, residual)  # [B, Channel, d_model]
        # BatchNorm1d expects [B, enc_in, d_model], normalize trên channel
        attn_out = attn_out + residual
        res_2 = self.channel_att_norm(attn_out)  # [B, Channel, d_model]
        ff_out = self.ffn(attn_out)
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

        # Shared Linear cho subsequence
        self.subseq_predictor = nn.Linear(self.period_len * self.d_model, self.period_len)

        # Linear Stream
        self.fc5 = nn.Linear(seq_len, pred_len * 2)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t):
        # s: [Batch, Input, Channel]
        B, I, C = s.shape
        s_proj = s.permute(0, 2, 1)  # [B, Channel, Input]
        s_proj = self.channel_proj(s_proj)  # [B, Channel, d_model]

        # Multi-layer Channel Attention (trên channel)
        for i in range(self.n_layers):
            s_proj = self.channel_attn_blocks[i](s_proj)  # [B, Channel, d_model]

        # Chia thành các subsequence theo period_len
        # s_proj: [B, Channel, d_model]
        s_proj = s_proj.reshape(B, C, self.seg_num_x, self.period_len, self.d_model)  # [B, C, seg_num_x, period_len, d_model]
        s_proj = s_proj.permute(0, 2, 1, 3, 4)  # [B, seg_num_x, C, period_len, d_model]

        # Dự đoán từng subsequence
        y_list = []
        for i in range(self.seg_num_x):
            subseq = s_proj[:, i]  # [B, C, period_len, d_model]
            # Attention cho từng subsequence (nếu muốn, có thể dùng một block attention nhỏ)
            subseq = subseq.reshape(B, C, -1)  # [B, C, period_len * d_model]
            y_sub = self.subseq_predictor(subseq)  # [B, C, period_len]
            y_list.append(y_sub)
        y = torch.cat(y_list, dim=-1)  # [B, C, seq_len]

        # Nếu muốn dự báo pred_len, có thể cắt hoặc interpolate y
        y = y[:, :, :self.pred_len]  # [B, C, pred_len]
        y = y.permute(0, 2, 1)      # [B, pred_len, C]

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