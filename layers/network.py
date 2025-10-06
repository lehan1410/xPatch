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

        # Seasonal Stream: Multihead Attention + Linear blocks
        self.channelAggregator = nn.MultiheadAttention(embed_dim=self.seq_len, num_heads=4, batch_first=True, dropout=0.5)
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

        # Linear Stream (giữ nguyên như cũ)
        self.fc5 = nn.Linear(seq_len, pred_len)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len)
        self.fc7 = nn.Linear(pred_len, pred_len // 2)
        self.fc8 = nn.Linear(pred_len // 2, pred_len)

    def forward(self, s, t):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        s = s.permute(0,2,1) # [Batch, Channel, Input]
        t = t.permute(0,2,1) # [Batch, Channel, Input]

        B, C, I = s.shape
        t = torch.reshape(t, (B*C, I))

        # Seasonal Stream: Multihead Attention + Linear blocks
        # Đầu vào cho MultiheadAttention: [Batch*Channel, seq_len] -> [Batch*Channel, 1, seq_len]
        s_attn_in = s.reshape(B*C, 1, I)  # [B*C, 1, seq_len]
        # MultiheadAttention expects [batch, seq_len, embed_dim], nhưng ở đây embed_dim=seq_len
        attn_out, _ = self.channelAggregator(s_attn_in, s_attn_in, s_attn_in)  # [B*C, 1, seq_len]
        attn_out = attn_out.squeeze(1)  # [B*C, seq_len]
        s_proj = self.input_proj(attn_out)  # [B*C, d_model]
        s_feat = self.model(s_proj)         # [B*C, d_model]
        y = self.output_proj(s_feat)        # [B*C, pred_len]
        y = y.reshape(B, C, self.pred_len).permute(0, 2, 1)  # [B, pred_len, C]

        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [B, pred_len, C]

        return t + y