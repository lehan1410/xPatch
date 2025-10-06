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

        # 1D convolution layer
        self.conv1d = nn.Conv1d(
            in_channels=self.enc_in, out_channels=self.enc_in,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False, groups=self.enc_in
        )

        # Average Pooling layer
        self.pool = nn.AvgPool1d(
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2
        )

        # Multihead Attention layer
        self.attention = nn.MultiheadAttention(embed_dim=self.enc_in, num_heads=self.enc_in, batch_first=True)

        # MLP layers for subsequence processing
        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.seg_num_y)
        )

        # Linear Stream (giữ nguyên)
        self.fc5 = nn.Linear(seq_len, pred_len)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len)
        self.fc7 = nn.Linear(pred_len, pred_len // 2)
        self.fc8 = nn.Linear(pred_len // 2, pred_len)

    def forward(self, s, t):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]

        # Step 1: Preprocess inputs
        s = s.permute(0, 2, 1)  # [Batch, Channel, Input]
        t = t.permute(0, 2, 1)  # [Batch, Channel, Input]

        B, C, I = s.shape
        t = torch.reshape(t, (B * C, I))

        # Step 2: Apply convolution and pooling to s
        s_conv = self.conv1d(s)  # [B, C, seq_len]
        s_pool = self.pool(s_conv)  # [B, C, seq_len]
        s = s_pool + s
        s = s.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)  # [B, period_len, seg_num_x]

        # Step 3: Apply attention on the subsequences
        s_attention, _ = self.attention(s, s, s)  # [B, period_len, seg_num_x]

        # Step 4: Process with MLP
        y = self.mlp(s_attention)
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1)  # [B, pred_len, enc_in]

        # Step 5: Linear Stream for t
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0, 2, 1)  # [B, pred_len, C]

        # Step 6: Combine the outputs from attention and linear stream
        return t + y
