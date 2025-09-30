import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model):
        super(Network, self).__init__()

        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in  = c_in
        self.period_len = period_len
        self.d_model = d_model

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        # Channel-independent Conv1d & Pooling
        self.conv1d = nn.Conv1d(
            in_channels=c_in, out_channels=c_in,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            groups=c_in,  # mỗi channel độc lập
            padding_mode="zeros", bias=False
        )
        self.pool = nn.AvgPool1d(
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2
        )

        # Channel-independent MLP (grouped linear)
        self.mlp1 = nn.Conv1d(
            in_channels=c_in, out_channels=c_in * self.d_model,
            kernel_size=1, groups=c_in, bias=True
        )
        self.act = nn.GELU()
        self.mlp2 = nn.Conv1d(
            in_channels=c_in * self.d_model, out_channels=c_in * self.seg_num_y,
            kernel_size=1, groups=c_in, bias=True
        )

        # Linear Stream (trend) - channel independence
        self.trend_regression = nn.Conv1d(
            in_channels=c_in, out_channels=c_in,
            kernel_size=seq_len, groups=c_in, bias=True
        )
        nn.init.constant_(self.trend_regression.weight, 1.0 / self.pred_len)
        nn.init.constant_(self.trend_regression.bias, 0.0)

    def forward(self, s, t):
        # s, t: [Batch, Input, Channel]
        B, L, C = s.shape

        # Seasonal Stream: channel independence
        s = s.permute(0, 2, 1)  # [B, C, L]
        s_conv = self.conv1d(s)
        s_pool = self.pool(s)
        s_concat = s_conv + s_pool + s  # [B, C, L]
        # Patch thành [B, C, seg_num_x, period_len]
        s_patch = s_concat.reshape(B, C, self.seg_num_x, self.period_len)
        s_patch = s_patch.mean(-1)  # [B, C, seg_num_x]
        y = self.mlp1(s_patch)
        y = self.act(y)
        y = self.mlp2(y)
        y = y.view(B, C, self.seg_num_y)
        y = y.permute(0, 2, 1)  # [B, pred_len//period_len, C]
        y = y.repeat_interleave(self.period_len, dim=1)[:, :self.pred_len, :]  # [B, pred_len, C]

        # Linear Stream: channel independence
        t = t.permute(0, 2, 1)  # [B, C, L]
        t_out = self.trend_regression(t)  # [B, C, 1]
        t_out = t_out.squeeze(-1).unsqueeze(1).repeat(1, self.pred_len, 1)  # [B, pred_len, C]

        return t_out + y