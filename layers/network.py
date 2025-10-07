import torch
from torch import nn

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

        # Conv1D và Pooling (giữ nguyên)
        self.conv1d = nn.Conv1d(
            in_channels=self.enc_in, out_channels=self.enc_in,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False, groups=self.enc_in
        )

        self.pool = nn.AvgPool1d(
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2
        )
        
        # Thêm attention mechanism
        # Đảm bảo num_heads chia hết cho period_len
        num_heads = 2
        if self.period_len % num_heads != 0:
            num_heads = 1
            
        self.segment_attention = nn.MultiheadAttention(
            embed_dim=self.period_len,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )

        # MLP vẫn giữ nguyên
        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.seg_num_y)
        )

        # Linear Stream
        self.fc5 = nn.Linear(seq_len, pred_len * 2)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        s = s.permute(0,2,1)  # [Batch, Channel, Input]
        t = t.permute(0,2,1)  # [Batch, Channel, Input]

        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        t = torch.reshape(t, (B*C, I))

        # Seasonal Stream: Conv1d + Pooling
        s_conv = self.conv1d(s)  # [B, C, seq_len]
        s_pool = self.pool(s_conv)  # [B, C, seq_len]
        s = s_pool + s
        
        # Reshape và phân đoạn dữ liệu
        s = s.reshape(-1, self.seg_num_x, self.period_len)  # [B*C, seg_num_x, period_len]
        
        # Áp dụng attention giữa các phân đoạn
        s_attn, _ = self.segment_attention(s, s, s)
        s = s + s_attn  # Residual connection
        
        # Tiếp tục xử lý như trước
        s = s.permute(0, 2, 1)  # [B*C, period_len, seg_num_x]
        y = self.mlp(s)
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1)

        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1)  # [Batch, Output, Channel] = [B, pred_len, C]

        return t + y