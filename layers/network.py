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

        # Attention cho channel
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=self.enc_in, num_heads=1, batch_first=True
        )

        # Linear pipeline cho seasonal
        self.input_proj = nn.Linear(self.seq_len, self.d_model)
        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
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
        t_trend = torch.reshape(t, (B*C, I))

        # Seasonal Stream: chỉ attention các channel
        s_channel = s.permute(0, 2, 1)  # [B, seq_len, C]
        channel_attn_out, _ = self.channel_attn(s_channel, s_channel, s_channel)  # [B, seq_len, C]
        s_channel = channel_attn_out.permute(0, 2, 1)  # [B, C, seq_len]

        # Linear pipeline cho seasonal
        seasonal = s_channel.reshape(B*C, self.seq_len)  # [B*C, seq_len]
        seasonal = self.input_proj(seasonal)             # [B*C, d_model]
        seasonal = self.model(seasonal)                  # [B*C, d_model]
        seasonal = self.output_proj(seasonal)            # [B*C, pred_len]
        seasonal = seasonal.reshape(B, C, self.pred_len)
        seasonal = seasonal.permute(0, 2, 1)             # [B, pred_len, C]

        # Trend Stream: thêm residual
        t_trend_origin = t_trend.clone()                 # [B*C, seq_len]
        t_trend = self.fc5(t_trend)
        t_trend = self.gelu1(t_trend)
        t_trend = self.ln1(t_trend)
        t_trend = self.fc7(t_trend)
        t_trend = self.fc8(t_trend)
        # Residual: cộng đầu vào gốc (sau khi resize cho đúng shape)
        t_trend = t_trend + t_trend_origin[:, :self.pred_len]
        t_trend = torch.reshape(t_trend, (B, C, self.pred_len))
        t_trend = t_trend.permute(0,2,1) # [B, pred_len, C]

        return t_trend + seasonal