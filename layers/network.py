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

        # Multihead Attention cho seasonal stream
        self.seasonal_proj = nn.Linear(self.period_len, self.d_model)
        self.mha = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=2, batch_first=True)
        self.seasonal_out = nn.Linear(self.d_model, self.pred_len)

        # Linear Stream
        self.fc5 = nn.Linear(seq_len, pred_len)  
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len)
        self.fc7 = nn.Linear(pred_len, pred_len // 2)  
        self.fc8 = nn.Linear(pred_len // 2, pred_len)
        self.trend_fc = nn.Linear(1, pred_len)

    def forward(self, s, t):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        s = s.permute(0,2,1) # [Batch, Channel, Input]
        t = t.permute(0,2,1) # [Batch, Channel, Input]

        B, C, I = s.shape
        # Chia thành các đoạn chu kỳ
        s_segs = s.reshape(B*C, self.seq_len // self.period_len, self.period_len)  # [B*C, seg_num_x, period_len]
        s_proj = self.seasonal_proj(s_segs)  # [B*C, seg_num_x, d_model]
        # Multihead Attention
        attn_out, _ = self.mha(s_proj, s_proj, s_proj)  # [B*C, seg_num_x, d_model]
        # Lấy đặc trưng tổng hợp
        seasonal_feat = attn_out.mean(dim=1)  # [B*C, d_model]
        y = self.seasonal_out(seasonal_feat)  # [B*C, pred_len]
        y = y.reshape(B, C, self.pred_len).permute(0,2,1)  # [B, pred_len, C]

        # Linear Stream
        t = torch.reshape(t, (B*C, I))
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [B, pred_len, C]

        global_trend = t.mean(dim=2, keepdim=True)  # [B, pred_len, 1]
        global_trend = self.trend_fc(global_trend)  # [B, pred_len, pred_len]
        global_trend = global_trend.mean(dim=2, keepdim=True)
        t = t + global_trend 

        return t + y