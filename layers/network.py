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

        self.conv1d = nn.Conv1d(
            in_channels=self.enc_in, out_channels=self.enc_in,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False, groups=self.enc_in
        )

        self.pool = nn.AvgPool1d(
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2
        )

        # Linear để trộn subsequence
        self.subseq_mixer = nn.Sequential(
            nn.Linear(self.seg_num_x, self.seg_num_x * 2),
            nn.GELU(),
            nn.Linear(self.seg_num_x * 2, self.seg_num_x)
        )

        # Linear để trộn channel
        self.channel_mixer = nn.Sequential(
            nn.Linear(self.enc_in, self.enc_in * 2),
            nn.GELU(),
            nn.Linear(self.enc_in * 2, self.enc_in)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.seg_num_y)
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
        s_conv = self.conv1d(s)  # [B, C, seq_len]
        s_pool = self.pool(s_conv)  # [B, C, seq_len]
        s = s_pool + s

        # Chia thành các subsequence
        s = s.reshape(B, C, self.seg_num_x, self.period_len)  # [B, C, seg_num_x, period_len]
        s = s.mean(-1)  # [B, C, seg_num_x]  # lấy trung bình mỗi subsequence

        # Trộn subsequence (theo chiều seg_num_x)
        s_subseq = self.subseq_mixer(s)  # [B, C, seg_num_x]

        # Trộn channel (theo chiều channel)
        s_channel = s.permute(0, 2, 1)  # [B, seg_num_x, C]
        s_channel = self.channel_mixer(s_channel)  # [B, seg_num_x, C]
        s_channel = s_channel.permute(0, 2, 1)  # [B, C, seg_num_x]

        # Cộng đặc trưng subsequence và channel
        s_fused = s_subseq + s_channel  # [B, C, seg_num_x]

        # Đưa qua MLP như cũ
        y = self.mlp(s_fused)  # [B, C, seg_num_y]
        y = y.reshape(B, C, self.seg_num_y * self.period_len)[:, :, :self.pred_len]  # [B, C, pred_len]
        y = y.permute(0, 2, 1)  # [B, pred_len, C]

        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [Batch, Output, Channel] = [B, pred_len, C]

        return t + y