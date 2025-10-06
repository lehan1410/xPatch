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

        # Thêm Conv1d cho channel independence
        self.conv1d = nn.Conv1d(
            in_channels=self.enc_in, out_channels=self.enc_in,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False, groups=self.enc_in
        )

        self.attn = nn.MultiheadAttention(embed_dim=self.period_len, num_heads=2, batch_first=True)

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

        # Thêm Conv1d trước khi chia subsequence
        s_conv = self.conv1d(s)  # [B, C, seq_len]
        s = s_conv + s

        # Chia thành các subsequence
        s = s.reshape(-1, self.seg_num_x, self.period_len)  # [B*C, seg_num_x, period_len]

        # Attention giữa các subsequence
        s_attn, _ = self.attn(s, s, s)
        s = s + s_attn

        s = s.permute(0, 2, 1)  # [B*C, period_len, seg_num_x]
        s = s.reshape(-1, self.seg_num_x)

        y = self.mlp(s)
        y = y.reshape(B, C, self.period_len, self.seg_num_y)
        y = y.permute(0, 1, 2, 3).reshape(B, C, self.pred_len)
        y = y.permute(0, 2, 1)  # [B, pred_len, C]

        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [Batch, pred_len, Channel]

        return t + y