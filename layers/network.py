import torch
from torch import nn

class MixerBlock(nn.Module):
    def __init__(self, channel, seq_len, d_model, dropout=0.1, expansion=2):
        super().__init__()
        self.norm = nn.LayerNorm(seq_len)
        self.mlp = nn.Sequential(
            nn.Linear(seq_len, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, seq_len)
        )

    def forward(self, x):
        # x: [B, C, seq_len]
        # Chỉ trộn thời gian cho từng channel
        x_norm = self.norm(x)  # [B, C, seq_len]
        # Đưa từng channel qua MLP thời gian
        z = self.mlp(x_norm)   # [B, C, seq_len]
        out = x + z            # residual thời gian
        return out

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

        self.activation = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(dropout),
        )

        self.channel_attn = nn.MultiheadAttention(
            embed_dim=self.enc_in, num_heads=1, batch_first=True
        )

        self.mixer = MixerBlock(channel=self.enc_in, seq_len=self.seq_len, d_model=self.d_model, dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.seg_num_y)
        )

        self.fc5 = nn.Linear(seq_len, pred_len * 2)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t):
        s = s.permute(0,2,1) # [B, C, Input]
        t = t.permute(0,2,1) # [B, C, Input]

        B, C, I = s.shape
        t = torch.reshape(t, (B*C, I))

        # Conv1d cho từng channel riêng biệt
        s_conv = self.conv1d(s.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len)
        s_pool1 = self.pool(s_conv)
        s_act = self.activation(s_pool1)
        s_feat = s_act + s  # residual

        # Attention channel
        s_attn_in = s_feat.permute(0, 2, 1)  # [B, seq_len, C]
        s_attn_out, _ = self.channel_attn(s_attn_in, s_attn_in, s_attn_in)
        s_attn_out = s_attn_out.permute(0, 2, 1)  # [B, C, seq_len]
        s_fusion = s_feat + s_attn_out  # residual

        # Mixer block cho các chuỗi thời gian
        s_mixed = self.mixer(s_fusion) + s  # [B, C, seq_len]

        # Reshape thành patch/subsequence
        s_patch = s_mixed.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # Đưa qua MLP để dự đoán
        y = self.mlp(s_patch)
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1)

        # Trend Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1)

        return t + y