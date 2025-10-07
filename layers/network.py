import torch
from torch import nn
import math

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

        # Thêm xử lý đặc trưng thời gian từ seq_x_mark (thường có 4-5 đặc trưng thời gian)
        self.time_features = 4  # month, day, weekday, hour (có thể thêm minute)
        self.time_embedding = nn.Linear(self.time_features, self.period_len)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.enc_in, out_channels=self.enc_in,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False, groups=1
        )

        self.pool = nn.AvgPool1d(
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2
        )
        
        # Segment attention with time-aware capability
        self.segment_attention = nn.MultiheadAttention(
            embed_dim=self.period_len,
            num_heads=2,
            batch_first=True
        )
        
        # Thêm layer để kết hợp thông tin thời gian với biểu diễn segment
        self.time_fusion = nn.Sequential(
            nn.Linear(self.period_len * 2, self.period_len),
            nn.GELU(),
            nn.Linear(self.period_len, self.period_len)
        )

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

    def forward(self, s, t, seq_x_mark):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        # seq_x_mark: [Batch, Input, TimeFeatures]
        
        s = s.permute(0,2,1) # [Batch, Channel, Input]
        t = t.permute(0,2,1) # [Batch, Channel, Input]

        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        t = torch.reshape(t, (B*C, I))

        # Seasonal Stream: Conv1d + Pooling
        s_conv = self.conv1d(s)  # [B, C, seq_len]
        s_pool = self.pool(s_conv)  # [B, C, seq_len]
        s = s_pool + s
        
        # Reshape cho phân đoạn
        s = s.reshape(-1, self.seg_num_x, self.period_len)  # [B*C, seg_num_x, period_len]
        
        # Xử lý thông tin thời gian từ seq_x_mark
        if seq_x_mark is not None:
            # Phân đoạn thời gian tương tự như dữ liệu
            time_features = seq_x_mark.shape[-1]  # Số lượng đặc trưng thời gian
            
            # Reshape seq_x_mark để khớp với phân đoạn
            seq_x_mark_segmented = seq_x_mark.reshape(B, self.seg_num_x, self.period_len, time_features)
            
            # Trung bình theo period_len để có đặc trưng cho mỗi phân đoạn
            seq_x_mark_avg = torch.mean(seq_x_mark_segmented, dim=2)  # [B, seg_num_x, time_features]
            
            # Tạo embedding cho mỗi phân đoạn thời gian
            time_embed = self.time_embedding(seq_x_mark_avg)  # [B, seg_num_x, period_len]
            
            # Mở rộng cho mỗi kênh
            time_embed = time_embed.unsqueeze(1).expand(-1, C, -1, -1)  # [B, C, seg_num_x, period_len]
            time_embed = time_embed.reshape(B*C, self.seg_num_x, self.period_len)  # [B*C, seg_num_x, period_len]
            
            # Kết hợp với biểu diễn segment
            combined = torch.cat([s, time_embed], dim=-1)  # [B*C, seg_num_x, period_len*2]
            s_with_time = self.time_fusion(combined.reshape(-1, self.period_len*2))
            s_with_time = s_with_time.reshape(-1, self.seg_num_x, self.period_len)
            
            # Sử dụng biểu diễn kết hợp
            s = s_with_time
        
        # Tiếp tục như bình thường
        s_attn, _ = self.segment_attention(s, s, s)
        s = s + s_attn  # Residual connection
        
        # Tiếp tục xử lý như cũ
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
        t = t.permute(0,2,1)

        return t + y