import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model, dropout=0.1):
        super(Network, self).__init__()

        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in  = c_in
        self.period_len = period_len
        self.d_model = d_model
        self.dropout = dropout

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

        self.activation = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
        )

        self.conv1d2 = nn.Conv1d(
            in_channels=self.enc_in, out_channels=self.enc_in,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False, groups=1
        )

        self.pool2 = nn.AvgPool1d(
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2
        )

        self.tanh = nn.Tanh()

        # Attention giữa các subsequence (patch)
        self.patch_attn = nn.MultiheadAttention(
            embed_dim=self.period_len, num_heads=2, batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.seg_num_y)
        )

        # Linear Stream cho trend
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

        B, C, I = s.shape
        t = torch.reshape(t, (B*C, I))

        # Block 1: Conv1d + Pool + Activation + Residual
        s_res1 = s
        s_conv1 = self.conv1d(s)                # [B, C, seq_len]
        s_pool1 = self.pool(s_conv1)            # [B, C, seq_len]
        s_act1 = self.activation(s_pool1)       # [B, C, seq_len]
        s_act1 = s_act1 + s_res1                # Residual connection

        # Block 2: Conv1d2 + Pool2 + Tanh + Residual
        s_res2 = s_act1
        s_conv2 = self.conv1d2(s_act1)          # [B, C, seq_len]
        s_pool2 = self.pool2(s_conv2)           # [B, C, seq_len]
        s_out = self.tanh(s_pool2)              # [B, C, seq_len]
        s_out = s_out + s_res2                  # Residual connection

        # Reshape thành patch/subsequence
        s_patch = s_out.reshape(B * C, self.seg_num_x, self.period_len)  # [B*C, patch_num, period_len]

        # Attention giữa các subsequence
        patch_attn_out, _ = self.patch_attn(s_patch, s_patch, s_patch)   # [B*C, patch_num, period_len]

        # Đưa qua MLP để dự đoán
        y = self.mlp(patch_attn_out)                                    # [B*C, period_len, seg_num_y]
        y = y.permute(0, 2, 1).reshape(B, C, self.seg_num_y * self.period_len)
        y = y[:, :, :self.pred_len]                                     # [B, C, pred_len]
        y = y.permute(0, 2, 1)                                          # [B, pred_len, C]

        # Trend Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1)

        return t + y