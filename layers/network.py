import torch
from torch import nn

class MLPMixerBlock(nn.Module):
    def __init__(self, period_len, num_period, hidden_dim=64):
        super().__init__()
        # Token-mixing MLP (trộn giữa các period)
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(num_period),  # Sửa lại thành period_len
            nn.Linear(num_period, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_period)
        )
        # Channel-mixing MLP (trộn giữa các giá trị trong period)
        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(period_len),  # Sửa lại thành num_period
            nn.Linear(period_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, period_len)
        )

    def forward(self, x):
        # x: [B, period_len, num_period]
        # Token-mixing: trộn theo chiều period (2)
        y = x + self.token_mlp(x)
        # Channel-mixing: trộn theo chiều channel (1)
        y = y + self.channel_mlp(y.transpose(1,2)).transpose(1,2)
        return y
class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, c_in):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in  = c_in
        self.period_len = 24
        self.d_model = 128

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False
        )
        self.pool = nn.AvgPool1d(
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2
        )

        # self.mixer = MLPMixerBlock(self.period_len, self.seg_num_x, hidden_dim=16)


        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.seg_num_y)
        )

        # Linear Stream
        self.linear_stream = nn.Sequential(
            nn.Linear(seq_len, seq_len // 2),
            nn.GELU(),
            nn.Linear(seq_len // 2, seq_len // 4),
            nn.GELU(),
            nn.Linear(seq_len // 4, pred_len * 2),
            nn.GELU(),
            nn.LayerNorm(pred_len * 2),
            nn.Linear(pred_len * 2, pred_len),
            nn.GELU(),
            nn.Linear(pred_len, pred_len)
        )

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
        s_conv = self.conv1d(s.reshape(-1, 1, self.seq_len))
        s_pool = self.pool(s.reshape(-1, 1, self.seq_len))
        s_concat = s_conv + s_pool
        s_concat = s_concat.reshape(-1, self.enc_in, self.seq_len) + s
        s = s_concat.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        # s = self.mixer(s)
        y = self.mlp(s)
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1) # [B, pred_len, enc_in]


        # Linear Stream
        t = self.linear_stream(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [Batch, Output, Channel] = [B, pred_len, C]

        return t + y