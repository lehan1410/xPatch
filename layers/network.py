import torch
from torch import nn
import math

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

        # Thêm Time Embeddings
        self.time_features = 4  # hour, day, weekday, month
        self.time_embedding = nn.Linear(self.time_features, self.d_model)
        
        # Thêm positional encoding
        self.pos_encoder = PositionalEncoding(self.period_len, dropout=0.1)
        
        # Conv1D và Pooling
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
        
        # Attention
        num_heads = 2
            
        self.segment_attention = nn.MultiheadAttention(
            embed_dim=self.period_len,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Normalization layer
        self.norm1 = nn.LayerNorm(self.period_len)
        
        # MLP cải tiến
        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model * 2, self.seg_num_y)
        )

        # Linear Stream
        self.fc5 = nn.Linear(seq_len, pred_len * 2)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t, seq_x_mark=None):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        # seq_x_mark: [Batch, Input, TimeFeatures]
        
        s = s.permute(0,2,1)  # [Batch, Channel, Input]
        t = t.permute(0,2,1)  # [Batch, Channel, Input]

        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        t = torch.reshape(t, (B*C, I))

        # Seasonal Stream
        s_conv = self.conv1d(s)  # [B, C, seq_len]
        s_pool = self.pool(s_conv)  # [B, C, seq_len]
        s = s_pool + s
        
        # Reshape
        s = s.reshape(-1, self.seg_num_x, self.period_len)  # [B*C, seg_num_x, period_len]
        
        # Thêm thông tin thời gian nếu có
        if seq_x_mark is not None:
            # Xử lý time features
            time_embed = self.time_embedding(seq_x_mark)  # [B, Input, d_model]
            
            # Reshape để phù hợp với dữ liệu phân đoạn
            time_embed = time_embed.reshape(B, self.seg_num_x, -1)
            
            # Mở rộng cho mỗi kênh và cắt để khớp period_len
            time_embed = time_embed.unsqueeze(1).expand(-1, C, -1, -1)  # [B, C, seg_num_x, d_model]
            time_embed = time_embed.reshape(B*C, self.seg_num_x, -1)  # [B*C, seg_num_x, d_model]
            
            # Inject time information (đơn giản hóa cách truyền thông tin thời gian)
            s = s + time_embed[:, :, :self.period_len]
        
        # Thêm positional encoding
        s = self.pos_encoder(s)
        
        # Attention
        s_attn, _ = self.segment_attention(s, s, s)
        s = s + s_attn  # Residual connection
        
        # Normalization
        s = self.norm1(s)
        
        # MLP
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)