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
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in  = c_in
        self.period_len = period_len
        self.d_model = d_model

        self.channel_proj = nn.Linear(self.seq_len, self.d_model)
        self.channel_attn_blocks = nn.ModuleList([
            channel_attn_block(self.enc_in, self.d_model, 0.1)
            for _ in range(2)
        ])

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

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

        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.seg_num_y)
        )

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

        # Seasonal Stream: Conv1d + Pooling
        s_conv = self.conv1d(s)  # [B, C, seq_len]
        s_pool = self.pool(s_conv)  # [B, C, seq_len]
        s_seasonal = s_pool + s
        # Attention branch (channel-wise)
        s_attn = self.channel_proj(s_seasonal) 
        for i in range(len(self.channel_attn_blocks)):
            s_attn = self.channel_attn_blocks[i](s_attn)  # [B, C, d_model]
        # Reduce d_model to match s_seasonal for fusion (e.g. mean or linear)
        attn_info = s_attn.mean(-1, keepdim=True)         # [B, C, 1]
        attn_info = attn_info.expand(-1, -1, self.period_len * self.seg_num_x)  # [B, C, seq_len] (if needed)

        # Fusion: add attention info to seasonal stream
        s_fused = s_seasonal + attn_info[:, :, :s_seasonal.shape[2]]  # [B, C, seq_len]

        # MLP branch
        s_mlp_in = s_fused.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        y = self.mlp(s_mlp_in)
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1)

        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [Batch, Output, Channel] = [B, pred_len, C]

        return t + y