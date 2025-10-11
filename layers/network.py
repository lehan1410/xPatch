import torch
from torch import nn


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=False, groups=1):
        super().__init__()
        pad_left = (kernel_size - 1) // 2
        pad_right = (kernel_size - 1) - pad_left
        self.pad = nn.ConstantPad1d((pad_left, pad_right), 0)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
            groups=groups
        )

    def forward(self, x):
        x = self.pad(x)
        return self.conv(x)

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

        self.conv1d = CausalConv1d(
            in_channels=self.enc_in, out_channels=self.enc_in,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False, groups=self.enc_in
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

        # Seasonal Stream: chỉ attention các channel
        s_conv = self.conv1d(s)  # [B, C, seq_len]
        # s_pool = self.pool(s_conv)  # [B, C, seq_len]
        s = s_conv + s
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