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

        # Attention giữa các channel tại mỗi bước thời gian
        self.channel_attn = nn.MultiheadAttention(embed_dim=self.enc_in, num_heads=1, batch_first=True)

        # Encode time features để làm query
        self.time_encoder = nn.Linear(time_feat_dim, self.enc_in)

        # MLP cho từng subsequence
        self.mlp = nn.Sequential(
            nn.Linear(self.enc_in, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.period_len)
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

        B, C, I = s.shape
        t = torch.reshape(t, (B*C, I))

        # Encode time features để làm query cho attention channel
        time_query = self.time_encoder(seq_x_mark)  # [B, seq_len, enc_in]

        # Attention giữa các channel tại mỗi bước thời gian
        attn_out, _ = self.channel_attn(
            query=time_query, key=s.permute(0,2,1), value=s.permute(0,2,1)
        )  # [B, seq_len, enc_in]

        # Chia thành các subsequence
        attn_out_subseq = attn_out.reshape(B, self.seg_num_x, self.period_len, self.enc_in)  # [B, seg_num_x, period_len, enc_in]

        # Lấy đặc trưng cho mỗi subsequence (mean theo period_len)
        subseq_feat = attn_out_subseq.mean(dim=2)  # [B, seg_num_x, enc_in]

        # Dự báo cho từng subsequence
        y = self.mlp(subseq_feat)  # [B, seg_num_x, period_len]
        y = y.reshape(B, self.seg_num_x * self.period_len, 1)  # [B, seq_len, 1]
        y = y.squeeze(-1)  # [B, seq_len]

        # Nếu muốn output shape [B, pred_len, C], có thể lấy slice hoặc reshape lại
        y = y[:, -self.pred_len:]  # [B, pred_len]

        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [Batch, pred_len, Channel]

        # Nếu y cần shape [B, pred_len, C], expand chiều cuối
        y = y.unsqueeze(-1).expand(-1, self.pred_len, C)  # [B, pred_len, C]

        return t + y