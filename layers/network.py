import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in  = c_in
        self.period_len = period_len
        self.d_model = d_model

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

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

        # Seasonal Stream MLP (channel independence)
        self.seasonal_fc1 = nn.Linear(seq_len, pred_len * 2)
        self.seasonal_gelu1 = nn.GELU()
        self.seasonal_ln1 = nn.LayerNorm(pred_len * 2)
        self.seasonal_fc2 = nn.Linear(pred_len * 2, pred_len)
        self.seasonal_fc3 = nn.Linear(pred_len, pred_len)

        # Linear Stream
        self.fc5 = nn.Linear(seq_len, pred_len * 2)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t):
        # s, t: [Batch, Input, Channel]
        s = s.permute(0,2,1) # [B, C, I]
        t = t.permute(0,2,1) # [B, C, I]

        B, C, I = s.shape

        # Seasonal Stream: channel independence
        s = s.reshape(B*C, I)
        s = self.seasonal_fc1(s)
        s = self.seasonal_gelu1(s)
        s = self.seasonal_ln1(s)
        s = self.seasonal_fc2(s)
        s = self.seasonal_fc3(s)
        y = s.reshape(B, C, self.pred_len).permute(0, 2, 1) # [B, pred_len, C]

        # Linear Stream: channel independence
        t = t.reshape(B*C, I)
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = t.reshape(B, C, self.pred_len).permute(0, 2, 1) # [B, pred_len, C]

        return t + y