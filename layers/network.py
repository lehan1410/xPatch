import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model, dropout=0.1):
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

        # Attention giữa các subsequence
        self.subseq_attn = nn.MultiheadAttention(
            embed_dim=self.period_len, num_heads=2, batch_first=True
        )

        self.channel_attn = nn.MultiheadAttention(
            embed_dim=self.enc_in, num_heads=1, batch_first=True
        )

        # FFT Layer tăng khả năng biểu diễn
        self.fft_layer = nn.Sequential(
            nn.Linear(self.period_len, int(self.period_len * 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(self.period_len * 2), self.period_len),
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
        s_channel = s.permute(0, 2, 1)  # [B, seq_len, C]
        channel_attn_out, _ = self.channel_attn(s_channel, s_channel, s_channel)  # [B, seq_len, C]
        s_channel = channel_attn_out.permute(0, 2, 1) # [B*C, seg_num_x, period_len]

        s_subseq = s_channel.reshape(-1, self.seg_num_x, self.period_len) # [B*C, seg_num_x, period_len]

        # Attention giữa các subsequence
        attn_out, _ = self.subseq_attn(s_subseq, s_subseq, s_subseq)  # [B*C, seg_num_x, period_len]

        # FFT Layer trên từng subsequence
        fft_out = self.fft_layer(attn_out)  # [B*C, seg_num_x, period_len]

        # Đưa vào MLP
        mlp_in = fft_out.permute(0, 2, 1) 

        # Đưa vào MLP
        y = self.mlp(mlp_in)
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1)

        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [Batch, Output, Channel] = [B, pred_len, C]

        return t + y