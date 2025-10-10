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

        self.num_subseq = self.pred_len // self.period_len

        # Attention cho channel
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=self.enc_in, num_heads=1, batch_first=True
        )

        # Attention theo chiều thời gian cho từng subsequence
        self.time_attn = nn.MultiheadAttention(
            embed_dim=self.period_len, num_heads=1, batch_first=True
        )

        # Linear pipeline cho seasonal từng subsequence
        self.input_proj = nn.Linear(self.seq_len, self.d_model)
        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU()
        )
        self.dropout_layer = nn.Dropout(self.dropout)
        # Linear để lấy đặc trưng cho subsequence (không dùng mean)
        self.subseq_fusion = nn.Linear(self.period_len * self.d_model, self.period_len)
        self.subseq_proj = nn.Linear(self.period_len, self.period_len)

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
        t_trend = torch.reshape(t, (B*C, I))

        # Seasonal Stream: attention các channel
        s_channel = s.permute(0, 2, 1)  # [B, seq_len, C]
        channel_attn_out, _ = self.channel_attn(s_channel, s_channel, s_channel)  # [B, seq_len, C]
        s_channel = channel_attn_out.permute(0, 2, 1)  # [B, C, seq_len]

        # Project lên d_model
        seasonal = self.input_proj(s_channel.reshape(B*C, self.seq_len))  # [B*C, d_model]
        seasonal = self.model(seasonal)                                   # [B*C, d_model]
        seasonal = self.dropout_layer(seasonal)

        # Dự đoán từng subsequence
        seasonal_subseq = []
        for i in range(self.num_subseq):
            # Tạo đặc trưng cho subsequence
            subseq_feat = seasonal.unsqueeze(1).repeat(1, self.period_len, 1)  # [B*C, period_len, d_model]
            # Attention theo chiều thời gian
            subseq_feat, _ = self.time_attn(subseq_feat, subseq_feat, subseq_feat)  # [B*C, period_len, d_model]
            # Lấy đặc trưng bằng Linear thay vì mean
            subseq_feat = subseq_feat.reshape(-1, self.period_len * self.d_model)  # [B*C, period_len * d_model]
            subseq_feat = self.subseq_fusion(subseq_feat)  # [B*C, period_len]
            # Dự đoán cho subsequence
            subseq_out = self.subseq_proj(subseq_feat)  # [B*C, period_len]
            seasonal_subseq.append(subseq_out)

        # Ghép các subsequence lại
        seasonal = torch.cat(seasonal_subseq, dim=-1)  # [B*C, pred_len]
        seasonal = seasonal.reshape(B, C, self.pred_len)
        seasonal = seasonal.permute(0, 2, 1)           # [B, pred_len, C]

        # Trend Stream: thêm residual
        t_trend_origin = t_trend.clone()               # [B*C, seq_len]
        t_trend = self.fc5(t_trend)
        t_trend = self.gelu1(t_trend)
        t_trend = self.ln1(t_trend)
        t_trend = self.fc7(t_trend)
        t_trend = self.fc8(t_trend)
        t_trend = t_trend + t_trend_origin[:, :self.pred_len]
        t_trend = torch.reshape(t_trend, (B, C, self.pred_len))
        t_trend = t_trend.permute(0,2,1) # [B, pred_len, C]

        return t_trend + seasonal