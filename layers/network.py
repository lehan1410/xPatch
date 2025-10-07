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

        # Encode time features cho từng subsequence
        self.time_encoder = nn.Linear(time_feat_dim, self.period_len)

        # Attention giữa các subsequence
        self.attn_subseq = nn.MultiheadAttention(embed_dim=self.period_len, num_heads=2, batch_first=True)

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

        # Conv1d
        s_conv = self.conv1d(s)  # [B, C, seq_len]
        s = s_conv + s           # [B, C, seq_len]

        # Chia thành các subsequence
        s_subseq = s.reshape(-1, self.seg_num_x, self.period_len)  # [B*C, seg_num_x, period_len]

        # Chia time features thành subsequence
        time_subseq = seq_x_mark.unsqueeze(1).repeat(1, C, 1, 1)  # [B, C, seq_len, time_feat_dim]
        time_subseq = time_subseq.reshape(-1, self.seg_num_x, self.period_len, seq_x_mark.shape[-1])  # [B*C, seg_num_x, period_len, time_feat_dim]
        # Lấy trung bình đặc trưng thời gian cho mỗi subsequence
        time_emb = time_subseq.mean(dim=2)  # [B*C, seg_num_x, time_feat_dim]
        # Encode thành embedding cùng chiều với period_len
        query = self.time_encoder(time_emb)  # [B*C, seg_num_x, period_len]

        # Attention giữa các subsequence, dùng time embedding làm query
        s_subseq_attn, _ = self.attn_subseq(query, s_subseq, s_subseq)  # [B*C, seg_num_x, period_len]

        s = s_subseq_attn

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