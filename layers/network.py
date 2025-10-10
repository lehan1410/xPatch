import torch
from torch import nn

class CausalConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=5, dropout=0.0):
        super(CausalConvBlock, self).__init__()
        module_list = [
            nn.ReplicationPad1d((kernel_size - 1, kernel_size - 1)),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size),
            nn.Tanh()
        ]
        self.causal_conv = nn.Sequential(*module_list)

    def forward(self, x):
        return self.causal_conv(x)

class MixerBlock(nn.Module):
    def __init__(self, num_subseq, period_len, num_channel, d_model, dropout=0.1, tfactor=2, dfactor=2):
        super().__init__()
        # Token mixing (trộn giữa các subsequence)
        self.token_mixer = nn.Sequential(
            nn.Linear(num_subseq, num_subseq * tfactor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_subseq * tfactor, num_subseq)
        )
        # Channel mixing (trộn giữa các channel)
        self.channel_mixer = nn.Sequential(
            nn.Linear(num_channel, num_channel * dfactor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_channel * dfactor, num_channel)
        )
        self.norm1 = nn.LayerNorm([period_len, num_subseq])
        self.norm2 = nn.LayerNorm([period_len, num_subseq])

    def forward(self, x):
        # x: [B, num_channel, period_len, num_subseq]
        x = self.norm1(x)
        # Token mixing: trộn giữa các subsequence
        # Đưa num_subseq về cuối
        x_token = x.permute(0, 1, 2, 3)  # [B, num_channel, period_len, num_subseq]
        x_token = x_token.reshape(-1, x_token.shape[-1])  # [B * num_channel * period_len, num_subseq]
        x_token = self.token_mixer(x_token)               # [B * num_channel * period_len, num_subseq]
        x_token = x_token.reshape(x.shape)                # [B, num_channel, period_len, num_subseq]
        x = x + x_token
        x = self.norm2(x)
        # Channel mixing: trộn giữa các channel
        # Đưa num_channel về cuối
        x_channel = x.permute(0, 3, 2, 1)                # [B, num_subseq, period_len, num_channel]
        x_channel = x_channel.reshape(-1, x_channel.shape[-1])  # [B * num_subseq * period_len, num_channel]
        x_channel = self.channel_mixer(x_channel)               # [B * num_subseq * period_len, num_channel]
        x_channel = x_channel.reshape(x.shape[0], x.shape[3], x.shape[2], x.shape[1])  # [B, num_subseq, period_len, num_channel]
        x_channel = x_channel.permute(0, 3, 2, 1)        # [B, num_channel, period_len, num_subseq]
        x = x + x_channel
        return x

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, num_channel, period_len, d_model, dropout=0.1, tfactor=2, dfactor=2):
        super(Network, self).__init__()

        self.pred_len = pred_len
        self.seq_len = seq_len
        self.num_channel = num_channel
        self.period_len = period_len
        self.d_model = d_model

        self.num_subseq_x = self.seq_len // self.period_len
        self.num_subseq_y = self.pred_len // self.period_len

        self.causal_conv = CausalConvBlock(d_model=self.num_channel, kernel_size=5, dropout=dropout)


        self.pool = nn.AvgPool1d(
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2
        )

        # Mixer block để trộn thông tin giữa subsequence và channel
        self.mixer = MixerBlock(
            num_subseq=self.num_subseq_x,
            period_len=self.period_len,
            num_channel=self.num_channel,
            d_model=self.d_model,
            dropout=dropout,
            tfactor=tfactor,
            dfactor=dfactor
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.num_subseq_x, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.num_subseq_y)
        )

        # Linear Stream
        self.fc5 = nn.Linear(seq_len, pred_len * 2)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t):
        # s: [Batch, Input, num_channel]
        # t: [Batch, Input, num_channel]
        s = s.permute(0,2,1) # [Batch, num_channel, Input]
        t = t.permute(0,2,1) # [Batch, num_channel, Input]

        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        t = torch.reshape(t, (B*C, I))

        # Seasonal Stream: Conv1d + Pooling
        s_conv = self.causal_conv(s)  # [B, num_channel, seq_len]
        s_pool = self.pool(s_conv)  # [B, num_channel, seq_len]
        s = s_pool + s

        # Chia thành các subsequence
        s_subseq = s.reshape(B, C, self.num_subseq_x, self.period_len)  # [B, num_channel, num_subseq, period_len]
        s_subseq = s_subseq.permute(0, 1, 3, 2)  # [B, num_channel, period_len, num_subseq]

        # Mixer block: trộn thông tin giữa subsequence và channel
        s_mixed = self.mixer(s_subseq)   # [B, num_channel, period_len, num_subseq]

        # Đưa về dạng [B*C, num_subseq, period_len] để vào MLP
        s_mixed = s_mixed.permute(0, 1, 3, 2).reshape(-1, self.num_subseq_x, self.period_len)
        y = self.mlp(s_mixed)
        y = y.permute(0, 2, 1).reshape(B, self.num_channel, self.pred_len)
        y = y.permute(0, 2, 1)

        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [Batch, Output, num_channel] = [B, pred_len, num_channel]

        return t + y