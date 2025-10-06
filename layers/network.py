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

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        # Attention giữa các segment
        self.segment_proj = nn.Linear(self.seg_num_x, self.d_model)
        self.attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=2, batch_first=True)
        self.segment_out = nn.Linear(self.d_model, self.seg_num_y)

        # Linear Stream (giữ nguyên)
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

        # Seasonal Stream: chia thành các segment nhỏ
        s_segs = s.reshape(B*C, self.seg_num_x, self.period_len)  # [B*C, seg_num_x, period_len]
        s_proj = self.segment_proj(s_segs)  # [B*C, seg_num_x, d_model]
        # Attention giữa các segment
        attn_out, _ = self.attn(s_proj, s_proj, s_proj)  # [B*C, seg_num_x, d_model]
        # Dự đoán từng segment output
        seg_outputs = self.segment_out(attn_out)  # [B*C, seg_num_x, period_len]
        # Ghép lại thành chuỗi cuối cùng
        y = seg_outputs.reshape(B, C, self.pred_len)
        y = y.permute(0,2,1)  # [B, pred_len, C]

        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [B, pred_len, C]

        return t + y