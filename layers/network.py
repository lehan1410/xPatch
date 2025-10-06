import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in  = c_in
        self.period_len = 24
        self.d_model = 128

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.input_proj = nn.Linear(self.period_len, self.d_model)
        self.mha = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=2, batch_first=True)
        self.seasonal_out = nn.Linear(self.d_model, self.period_len)

        # Linear Stream
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

        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        t = torch.reshape(t, (B*C, I))

        # Seasonal Stream: Conv1d + Pooling
        s_segs = s.reshape(B*C, self.seg_num_x, self.period_len)  # [B*C, seg_num_x, period_len]
        s_proj = self.input_proj(s_segs)  # [B*C, seg_num_x, d_model]
        attn_out, _ = self.mha(s_proj, s_proj, s_proj)  # [B*C, seg_num_x, d_model]
        seasonal_feat = attn_out.mean(dim=1)  # [B*C, d_model]
        y = self.seasonal_out(seasonal_feat)  # [B*C, period_len]
        y = y.repeat(1, self.seg_num_y)  # [B*C, pred_len]
        y = y.reshape(B, C, self.pred_len).permute(0,2,1)

        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [Batch, Output, Channel] = [B, pred_len, C]

        return t + y