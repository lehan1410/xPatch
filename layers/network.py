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

        self.conv1d = nn.ModuleList([
                nn.Conv1d(
                    in_channels=1, out_channels=1,
                    kernel_size=1 + 2 * (self.period_len // 2),
                    stride=1, padding=self.period_len // 2,
                    padding_mode="zeros", bias=False
                ) for _ in range(c_in)
            ])

        self.mlp = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.seg_num_x, self.d_model),
                    nn.GELU(),
                    nn.Linear(self.d_model, self.seg_num_y)
                ) for _ in range(c_in)
            ])

        # Linear Stream
        self.simple_trend_linear = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(c_in)
        ])

    def forward(self, s, t):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        # s = s.permute(0,2,1) # [Batch, Channel, Input]
        # t = t.permute(0,2,1) # [Batch, Channel, Input]

        B, L, C = s.shape
        # t = torch.reshape(t, (B*C, I))

        y = torch.zeros(B, self.pred_len, C, device=s.device, dtype=s.dtype)
        t_out = torch.zeros(B, self.pred_len, C, device=s.device, dtype=s.dtype)
        for i in range(C):
            # Seasonal stream cho từng channel
            s_i = s[:,:,i].unsqueeze(1) # [B, 1, L]
            s_conv = self.conv1d[i](s_i)
            s_concat = s_conv + s_i
            s_feat = s_concat.reshape(B, self.seg_num_x, self.period_len).permute(0,2,1)
            y_i = self.mlp[i](s_feat)
            y[:, :, i] = y_i.permute(0,2,1).reshape(B, self.pred_len)

            # Trend stream cho từng channel
            t_i = t[:,:,i]
            t_i = self.simple_trend_linear[i](t_i)
            t_out[:,:,i] = t_i
        return t_out + y