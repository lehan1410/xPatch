import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model, n_heads=4):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in  = c_in
        self.period_len = 24
        self.d_model = 128
        self.n_heads = n_heads  # Số lượng đầu của Multihead Attention

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        # Conv1d và Pooling
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

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.seg_num_y)
        )

        # Lớp Attention
        self.attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_heads)

        # Chuyển đổi kích thước của t (đảm bảo kích thước là d_model)
        self.linear_t = nn.Linear(self.enc_in, self.d_model)

        # Linear Stream
        self.fc5 = nn.Linear(seq_len, pred_len * 2)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        s = s.permute(0,2,1)  # [Batch, Channel, Input]
        t = t.permute(0,2,1)  # [Batch, Channel, Input]

        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        t = torch.reshape(t, (B*C, I))

        # Seasonal Stream: Conv1d + Pooling
        s_conv = self.conv1d(s)  # [B, C, seq_len]
        s_pool = self.pool(s_conv)  # [B, C, seq_len]
        s = s_pool + s
        s = s.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        y = self.mlp(s)
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1)

        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)

        # Sử dụng Linear để thay đổi kích thước của t thành d_model
        t = self.linear_t(t)  # Đảm bảo t có kích thước d_model
        t = torch.reshape(t, (B, self.d_model, self.pred_len))
        t = t.permute(0,2,1)  # [Batch, Output, Channel] = [B, pred_len, C]

        # Thêm Attention vào Linear Stream
        t_attn, _ = self.attn(t, t, t)  # [B, pred_len, C]
        t = t + t_attn  # Kết hợp với đầu ra của Attention

        return t + y
