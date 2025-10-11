import torch
from torch import nn


class CausalConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=5, dropout=0.0):
        super(CausalConvBlock, self).__init__()
        module_list = [
            nn.ReplicationPad1d((kernel_size - 1, kernel_size - 1)),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size),
            nn.Tanh()
        ]
        self.causal_conv = nn.Sequential(*module_list)

    def forward(self, x):
        return self.causal_conv(x)

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model, dropout=0.1):
        super(Network, self).__init__()

        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in  = c_in
        self.period_len = period_len
        self.d_model = d_model
        self.dropout = dropout

        # Attention cho channel
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=self.enc_in, num_heads=1, batch_first=True
        )

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = CausalConvBlock(
            d_model=self.enc_in,
            kernel_size=1 + 2 * (self.period_len // 2),
            dropout=self.dropout, padding=self.period_len // 2
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


        # Linear Stream cho trend
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

        B, C, I = s.shape
        t = torch.reshape(t, (B*C, I))

        # s_attn_in = s.permute(0, 2, 1)  # [B, Input, Channel]
        # s_attn_out, _ = self.channel_attn(s_attn_in, s_attn_in, s_attn_in)  # [B, Input, Channel]
        # s = s_attn_out.permute(0, 2, 1)

        # Seasonal Stream: chỉ attention các channel
        # s_conv = self.conv1d(s.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + s

        s_conv = self.conv1d(s)  # [B, C, seq_len]
        s_pool = self.pool(s_conv)  # [B, C, seq_len]
        s = s_pool + s
        s = s.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        y = self.mlp(s)
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1)

        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1)

        return t + y