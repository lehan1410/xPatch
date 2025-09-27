import torch
from torch import nn


class PeriodSEBlock(nn.Module):
    def __init__(self, period_len, num_period, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # Pool trên chiều num_period
        self.fc = nn.Sequential(
            nn.Linear(period_len, period_len // reduction, bias=False),
            nn.GELU(),
            nn.Linear(period_len // reduction, period_len, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, period_len, num_period]
        w = self.avg_pool(x)      # [B, period_len, 1]
        w = w.squeeze(-1)         # [B, period_len]
        w = self.fc(w)            # [B, period_len]
        w = w.unsqueeze(-1)       # [B, period_len, 1]
        return x * w 
    
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

        self.period_se = PeriodSEBlock(self.period_len, self.seg_num_x, reduction=4)

        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.seg_num_y)
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
        s_conv = self.conv1d(s.reshape(-1, 1, self.seq_len))
        s_pool = self.pool(s.reshape(-1, 1, self.seq_len))
        s_concat = s_conv + s_pool
        s_concat = s_concat.reshape(-1, self.enc_in, self.seq_len) + s
        s = s_concat.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)  # [B, period_len, num_period]

        # Period interaction block (nhẹ, chia sẻ trọng số)
        s = self.period_se(s)  # [B, period_len, num_period]

        y = self.mlp(s)
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1) # [B, pred_len, enc_in]

        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [Batch, Output, Channel] = [B, pred_len, C]

        return t + y