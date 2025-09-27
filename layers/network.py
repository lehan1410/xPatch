import torch
from torch import nn

class MLPMixerBlock(nn.Module):
    def __init__(self, period_len, num_period, hidden_dim=64):
        super().__init__()
        # Token-mixing MLP (trộn giữa các period)
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(num_period),
            nn.Linear(num_period, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_period)
        )
        # Channel-mixing MLP (trộn giữa các giá trị trong period)
        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(period_len),
            nn.Linear(period_len, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, period_len)
        )

    def forward(self, x):
        # x: [B, period_len, num_period]
        y = x + self.token_mlp(x.transpose(1,2)).transpose(1,2)
        y = y + self.channel_mlp(y)
        return y

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, c_in):
        super(Network, self).__init__()

        # ...existing code...
        self.period_len = 24
        self.seg_num_x = seq_len // self.period_len
        self.seg_num_y = pred_len // self.period_len

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

        # MLP-Mixer block
        self.mixer = MLPMixerBlock(self.period_len, self.seg_num_x, hidden_dim=64)

        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.seg_num_y)
        )

        # ...linear stream as before...

    def forward(self, s, t):
        # ...existing code...
        s_conv = self.conv1d(s.reshape(-1, 1, self.seq_len))
        s_pool = self.pool(s.reshape(-1, 1, self.seq_len))
        s_concat = s_conv + s_pool
        s_concat = s_concat.reshape(-1, self.enc_in, self.seq_len) + s
        s = s_concat.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)  # [B, period_len, num_period]

        # MLP-Mixer block
        s = self.mixer(s)  # [B, period_len, num_period]

        y = self.mlp(s)
        y = y.permute(0, 2, 1).reshape(s.shape[0], self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1) # [B, pred_len, enc_in]

        # ...linear stream as before...
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.avgpool1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (s.shape[0], self.enc_in, self.pred_len))
        t = t.permute(0,2,1)

        return t + y