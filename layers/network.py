import torch
from torch import nn

class PeriodGLUBlock(nn.Module):
    def __init__(self, period_len, num_period):
        super().__init__()
        self.linear = nn.Linear(num_period, num_period * 2)

    def forward(self, x):
        # x: [B, period_len, num_period]
        out = self.linear(x)  # [B, period_len, num_period*2]
        a, b = out.chunk(2, dim=-1)
        return a * torch.sigmoid(b)

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

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm([self.period_len, self.seg_num_x])

        self.period_glu = PeriodGLUBlock(self.period_len, self.seg_num_x)

        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.seg_num_y)
        )

        # Linear Stream
        self.trend_pool = nn.AdaptiveAvgPool1d(1)
        self.linear_stream = nn.Sequential(
            nn.LayerNorm(seq_len),
            nn.Linear(seq_len, seq_len // 2),
            nn.GELU(),
            nn.LayerNorm(seq_len // 2),
            nn.Linear(seq_len // 2, self.pred_len),
            nn.GELU(),
            nn.LayerNorm(self.pred_len)
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
        s_conv = self.conv1d(s.reshape(-1, 1, self.seq_len))
        s_pool = self.pool(s.reshape(-1, 1, self.seq_len))
        s_concat = s_conv + s_pool
        s_concat = s_concat.reshape(-1, self.enc_in, self.seq_len) + s

        global_feat = self.global_pool(s_concat).expand_as(s_concat)
        s_concat = s_concat + global_feat

        s = s_concat.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)  # [B, period_len, num_period]
        s = self.norm(s)
        s = self.period_glu(s)  # GLU block
        y = self.mlp(s)
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1)

        pooled = self.trend_pool(t.unsqueeze(1)).squeeze(-1)  # [B*C, 1]
        trend = self.linear_stream(t)                         # [B*C, pred_len]
        # Cộng thông tin global pooling vào từng bước dự báo
        trend = trend + pooled.expand_as(trend)
        trend = trend.reshape(B, C, self.pred_len)
        t = trend.permute(0, 2, 1)

        return t + y