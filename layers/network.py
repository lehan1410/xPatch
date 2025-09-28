import torch
from torch import nn

class ResidualMLP(nn.Module):
    def __init__(self, seg_num_x, d_model, seg_num_y):
        super().__init__()
        self.linear1 = nn.Linear(seg_num_x, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(d_model, seg_num_y)
        
        # Residual connection nếu input/output size khớp
        self.use_residual = (seg_num_x == seg_num_y)
        if not self.use_residual:
            self.proj = nn.Linear(seg_num_x, seg_num_y)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.ln1(out)
        out = self.gelu(out)
        out = self.linear2(out)
        
        if self.use_residual:
            return out + residual
        else:
            return out + self.proj(residual)

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in  = c_in
        self.period_len = period_len
        self.d_model = d_model

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

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

        # Thay thế MLP bằng ResidualMLP
        self.mlp = ResidualMLP(self.seg_num_x, self.d_model, self.seg_num_y)

        # Linear Stream
        self.fc5 = nn.Linear(seq_len, pred_len * 2)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        self.fc8 = nn.Linear(pred_len, pred_len)

        # Cross-stream enhancement
        self.cross_connection = nn.Linear(self.pred_len, self.pred_len)

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
        s_conv = self.conv1d(s.reshape(-1, 1, self.seq_len))
        s_pool = self.pool(s.reshape(-1, 1, self.seq_len))
        s_concat = s_conv + s_pool
        s_concat = s_concat.reshape(-1, self.enc_in, self.seq_len) + s
        s = s_concat.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        y = self.mlp(s)  # Sử dụng ResidualMLP
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1) # [B, pred_len, enc_in]

        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1)

        # Cross-stream enhancement
        t_enhanced = self.cross_connection(t.transpose(-2, -1)).transpose(-2, -1)
        y_enhanced = self.cross_connection(y.transpose(-2, -1)).transpose(-2, -1)
        
        # Kết hợp với trọng số nhỏ để tránh overfitting
        return (t + 0.1 * y_enhanced) + (y + 0.1 * t_enhanced)