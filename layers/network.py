import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model, time_feat_dim=4):
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

        # Encode time features để làm query cho attention channel
        self.time_encoder = nn.Linear(time_feat_dim, self.enc_in)
        # Attention giữa các channel tại mỗi bước thời gian
        self.channel_attn = nn.MultiheadAttention(embed_dim=self.enc_in, num_heads=1, batch_first=True)

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

    def forward(self, s, t, seq_x_mark):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        # seq_x_mark: [Batch, seq_len, time_feat_dim]
        s = s.permute(0,2,1) # [Batch, Channel, Input]
        t = t.permute(0,2,1) # [Batch, Channel, Input]

        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        t = torch.reshape(t, (B*C, I))

        # Seasonal Stream: Conv1d + Pooling
        s_conv = self.conv1d(s)  # [B, C, seq_len]
        s_pool = self.pool(s_conv)  # [B, C, seq_len]
        s = s_pool + s  # [B, C, seq_len]

        # Attention giữa các channel tại từng bước thời gian
        s_time = s.permute(0, 2, 1)  # [B, seq_len, C]
        time_query = self.time_encoder(seq_x_mark)  # [B, seq_len, C]
        attn_out, _ = self.channel_attn(
            query=time_query, key=s_time, value=s_time
        )  # [B, seq_len, C]

        # Chia thành các subsequence
        attn_out_subseq = attn_out.reshape(B, self.seg_num_x, self.period_len, C)  # [B, seg_num_x, period_len, C]
        # Lấy đặc trưng cho mỗi subsequence (mean theo period_len)
        subseq_feat = attn_out_subseq.mean(dim=2)  # [B, seg_num_x, C]
        # Dự báo cho từng subsequence
        y = self.mlp(subseq_feat.permute(0,2,1))  # [B, C, seg_num_y]
        y = torch.nn.functional.interpolate(y, size=self.pred_len, mode='linear', align_corners=False)  # [B, C, pred_len]
        y = y.permute(0,2,1)  # [B, pred_len, C]

        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [Batch, pred_len, Channel]

        return t + y