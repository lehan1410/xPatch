import torch
from torch import nn

class TokenMixer(nn.Module):
    def __init__(self, input_seq, pred_seq, dropout, factor):
        super(TokenMixer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_seq, pred_seq * factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_seq * factor, pred_seq)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = x.transpose(1, 2)
        return x

class Mixer(nn.Module):
    def __init__(self, input_seq, out_seq, channel, d_model, dropout, tfactor, dfactor):
        super(Mixer, self).__init__()
        self.tMixer = TokenMixer(input_seq, out_seq, dropout, tfactor)
        self.dropoutLayer = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm2d(channel)
        self.norm2 = nn.BatchNorm2d(channel)
        self.embeddingMixer = nn.Sequential(
            nn.Linear(d_model, d_model * dfactor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * dfactor, d_model)
        )

    def forward(self, x):
        # x: [Batch, Channel, Patch_number, d_model]
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)  # [B, d_model, C, Patch_number]
        x = self.dropoutLayer(self.tMixer(x))
        x = x.permute(0, 2, 3, 1)  # [B, C, Patch_number, d_model]
        x = self.norm2(x)
        x = x + self.dropoutLayer(self.embeddingMixer(x))
        return x

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model, dropout=0.1, tfactor=2, dfactor=2):
        super(Network, self).__init__()

        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in  = c_in
        self.period_len = period_len
        self.d_model = d_model
        self.dropout = dropout

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        # Attention cho channel
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=self.enc_in, num_heads=1, batch_first=True
        )

        # Conv1d để học pattern thời gian
        self.conv1d = nn.Conv1d(
            in_channels=self.enc_in, out_channels=self.enc_in,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False, groups=self.enc_in
        )

        # Project patch lên d_model
        self.patch_proj = nn.Linear(self.period_len, self.d_model)

        # Mixer block
        self.mixer = Mixer(
            input_seq=self.seg_num_x,
            out_seq=self.seg_num_y,
            channel=self.enc_in,
            d_model=self.d_model,
            dropout=self.dropout,
            tfactor=tfactor,
            dfactor=dfactor
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

        # Attention các channel
        s_channel = s.permute(0, 2, 1)  # [B, seq_len, C]
        channel_attn_out, _ = self.channel_attn(s_channel, s_channel, s_channel)  # [B, seq_len, C]
        s_channel = channel_attn_out.permute(0, 2, 1)  # [B, C, seq_len]

        # Conv1d để học pattern thời gian
        s_conv = self.conv1d(s_channel)  # [B, C, seq_len]

        # Chia thành các patch/subsequence
        s_patch = s_conv.reshape(B, C, self.seg_num_x, self.period_len)  # [B, C, seg_num_x, period_len]
        s_patch = self.patch_proj(s_patch)  # [B, C, seg_num_x, d_model]

        # Mixer block
        s_mixed = self.mixer(s_patch)  # [B, C, seg_num_y, d_model]

        # Flatten để ra dự báo
        y = s_mixed.reshape(B, C, self.seg_num_y * self.d_model)
        y = y[:, :, :self.pred_len]  # [B, C, pred_len]
        y = y.permute(0, 2, 1)      # [B, pred_len, C]

        # Trend Stream: thêm residual
        t_trend_origin = t.clone()
        t_trend = self.fc5(t)
        t_trend = self.gelu1(t_trend)
        t_trend = self.ln1(t_trend)
        t_trend = self.fc7(t_trend)
        t_trend = self.fc8(t_trend)
        t_trend = t_trend + t_trend_origin[:, :self.pred_len]
        t_trend = torch.reshape(t_trend, (B, C, self.pred_len))
        t_trend = t_trend.permute(0,2,1)

        return t_trend + y