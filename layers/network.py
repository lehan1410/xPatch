import torch
from torch import nn


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel * 2, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, L]
        b, c, l = x.size()
        y_avg = self.avg_pool(x).view(b, c)  # [B, C]
        y_max = self.max_pool(x).view(b, c)  # [B, C]
        y = torch.cat([y_avg, y_max], dim=1) # [B, 2C]
        y = self.fc(y).view(b, c, 1)         # [B, C, 1]
        return x * y.expand_as(x)
    
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

        self.temporal_emb = nn.Parameter(torch.zeros(self.seq_len, self.enc_in), requires_grad=True)
        nn.init.xavier_uniform_(self.temporal_emb)

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

        self.se_block = SEBlock(self.enc_in, reduction=8)

        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.seg_num_y),
        )

        # Linear Stream
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
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

        s = s + self.temporal_emb.T.unsqueeze(0)

        # Seasonal Stream: Conv1d + Pooling
        s_conv = self.conv1d(s.reshape(-1, 1, self.seq_len))
        s_pool = self.pool(s.reshape(-1, 1, self.seq_len))
        s_concat = s_conv + s_pool
        s_concat = s_concat.reshape(-1, self.enc_in, self.seq_len) + s
        s_se = self.se_block(s_concat)
        s = s_se.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
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