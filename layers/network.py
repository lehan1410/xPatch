import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, c_in, time_embed_dim=4):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in  = c_in
        self.period_len = 24
        self.d_model = 128
        self.time_embed_dim = time_embed_dim

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        # Learnable time embeddings
        self.hour_embed = nn.Embedding(24, time_embed_dim)
        self.weekday_embed = nn.Embedding(7, time_embed_dim)
        self.input_dim = c_in + 2 * time_embed_dim

        self.conv1d = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False
        )
        self.pool = nn.AvgPool1d(
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.seg_num_y)
        )

        # Linear Stream
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.gelu1 = nn.GELU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t, hour_of_day=None, weekday=None):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        # hour_of_day, weekday: [Batch, Input] (int tensor)

        # Add time embedding if provided
        if hour_of_day is not None and weekday is not None:
            hour_emb = self.hour_embed(hour_of_day)      # [B, Input, time_embed_dim]
            weekday_emb = self.weekday_embed(weekday)    # [B, Input, time_embed_dim]
            s = torch.cat([s, hour_emb, weekday_emb], dim=-1)  # [B, Input, C+2*D]
            c_in = self.input_dim
        else:
            c_in = self.enc_in

        s = s.permute(0,2,1) # [Batch, Channel, Input]
        t = t.permute(0,2,1) # [Batch, Channel, Input]

        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        t = torch.reshape(t, (B*C, I))

        # Seasonal Stream: Conv1d + Pooling
        s_conv = self.conv1d(s.reshape(-1, 1, self.seq_len))
        s_pool = self.pool(s.reshape(-1, 1, self.seq_len))
        s_concat = s_conv + s_pool
        s_concat = s_concat.reshape(-1, c_in, self.seq_len) + s
        s = s_concat.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        y = self.mlp(s)
        y = y.permute(0, 2, 1).reshape(B, c_in, self.pred_len)
        y = y.permute(0, 2, 1) # [B, pred_len, c_in]

        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.avgpool1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [Batch, Output, Channel] = [B, pred_len, C]

        # Nếu dùng time embedding, chỉ lấy các channel đầu ra tương ứng với biến gốc
        if c_in > self.enc_in:
            y = y[..., :self.enc_in]
            t = t[..., :self.enc_in]

        return t + y