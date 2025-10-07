import torch
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in = c_in
        self.period_len = period_len
        self.d_model = d_model

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        # Seasonal Stream - Depthwise để giữ tính độc lập kênh
        self.conv1d = nn.Conv1d(
            in_channels=self.enc_in, out_channels=self.enc_in,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False, groups=self.enc_in
        )

        # Attention cho phân đoạn - giữ lại nhưng đơn giản hóa
        self.segment_attention = nn.MultiheadAttention(
            embed_dim=self.period_len,
            num_heads=2,
            batch_first=True
        )
        
        # MLP đơn giản hóa
        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.seg_num_y)
        )

        # Linear Stream đơn giản hóa
        self.linear = nn.Sequential(
            nn.Linear(self.seq_len, self.pred_len * 2),
            nn.GELU(),
            nn.LayerNorm(self.pred_len * 2),
            nn.Linear(self.pred_len * 2, self.pred_len)
        )

    def forward(self, s, t, seq_x_mark=None):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        s = s.permute(0, 2, 1)  # [Batch, Channel, Input]
        t = t.permute(0, 2, 1)  # [Batch, Channel, Input]

        B, C, I = s.shape

        # Seasonal Stream: Conv1d để nắm bắt mẫu mùa vụ
        s_conv = self.conv1d(s)  # [B, C, seq_len]
        s = s_conv + s  # Residual connection
        
        # Reshape để xử lý segment
        s = s.reshape(B*C, self.seg_num_x, self.period_len)  # [B*C, seg_num_x, period_len]
        
        # Giữ lại attention giữa các phân đoạn
        s_attn, _ = self.segment_attention(s, s, s)
        s = s + s_attn  # Residual connection
        
        # Hoán vị để xử lý bằng MLP
        s = s.permute(0, 2, 1)  # [B*C, period_len, seg_num_x]
        y = self.mlp(s)  # [B*C, period_len, seg_num_y]
        y = y.permute(0, 2, 1)  # [B*C, seg_num_y, period_len]
        
        # Reshape lại kết quả
        y = y.reshape(B, C, self.pred_len)
        y = y.permute(0, 2, 1)  # [B, pred_len, C]
        
        # Linear Stream xử lý đơn giản hơn
        t_flat = t.reshape(B*C, -1)  # [B*C, seq_len]
        y_linear = self.linear(t_flat)  # [B*C, pred_len]
        y_linear = y_linear.reshape(B, C, -1)  # [B, C, pred_len]
        y_linear = y_linear.permute(0, 2, 1)  # [B, pred_len, C]
        
        return y_linear + y