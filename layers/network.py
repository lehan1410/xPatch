import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model, num_scales=3):
        super(Network, self).__init__()

        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in  = c_in
        self.period_len = period_len
        self.d_model = d_model
        self.num_scales = num_scales

        # MultiScale Attention
        self.attn_proj = nn.Linear(self.enc_in, self.d_model)
        self.time_proj = nn.Linear(period_len, self.d_model)
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(self.d_model, num_heads=4, batch_first=True)
            for _ in range(num_scales)
        ])
        self.out_proj = nn.Linear(self.d_model, self.enc_in)

        # Linear Stream
        self.fc5 = nn.Linear(seq_len, pred_len * 2)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t, seq_x_mark):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        s = s.permute(0,2,1) # [Batch, Channel, Input]
        t = t.permute(0,2,1) # [Batch, Channel, Input]

        B, C, I = s.shape
        t = torch.reshape(t, (B*C, I))

        # Multiscale Attention Stream
        attn_outputs = []
        for scale in range(self.num_scales):
            factor = 2 ** scale
            s_ds = s[:, :, ::factor]  # [B, C, I//factor]
            time_ds = seq_x_mark[:, ::factor, :]  # [B, I//factor, period_len]
            s_proj = self.attn_proj(s_ds.permute(0,2,1))  # [B, I//factor, d_model]
            # Reshape time_ds for Linear
            B_ds, I_ds, P = time_ds.shape
            time_ds_reshape = time_ds.reshape(-1, P)  # [B*I//factor, period_len]
            time_emb = self.time_proj(time_ds_reshape) # [B*I//factor, d_model]
            time_emb = time_emb.reshape(B_ds, I_ds, self.d_model) # [B, I//factor, d_model]
            attn_out, _ = self.attn_layers[scale](s_proj, time_emb, time_emb)
            attn_out = self.out_proj(attn_out)            # [B, I//factor, enc_in]
            # Upsample to pred_len
            attn_out = attn_out.permute(0,2,1)
            attn_out = nn.functional.interpolate(attn_out, size=self.pred_len, mode='linear', align_corners=False)
            attn_outputs.append(attn_out)

        # Tổng hợp các attention outputs từ các scale
        y = sum(attn_outputs) / self.num_scales  # [B, enc_in, pred_len]
        y = y.permute(0,2,1)  # [B, pred_len, enc_in]

        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [Batch, Output, Channel] = [B, pred_len, C]

        return t + y