import torch
from torch import nn
import math

class SeasonalPatternAttention(nn.Module):
    def __init__(self, period_len, num_patterns=4):
        super().__init__()
        
        # Các mẫu mùa vụ điển hình có thể học được
        self.patterns = nn.Parameter(torch.randn(num_patterns, period_len))
        
        # Projection để tính toán attention scores
        self.query_proj = nn.Linear(period_len, period_len)
        self.key_proj = nn.Linear(period_len, period_len)
        self.value_proj = nn.Linear(period_len, period_len)
        
    def forward(self, x):
        # x: [B*C, seg_num_x, period_len]
        
        batch_size, seg_num = x.shape[0], x.shape[1]
        
        # Mở rộng patterns cho tất cả batch và segments
        patterns = self.patterns.unsqueeze(0).unsqueeze(0)  # [1, 1, num_patterns, period_len]
        patterns = patterns.expand(batch_size, seg_num, -1, -1)  # [B*C, seg_num_x, num_patterns, period_len]
        
        # Tính toán attention scores
        queries = self.query_proj(x).unsqueeze(2)  # [B*C, seg_num_x, 1, period_len]
        keys = self.key_proj(patterns)  # [B*C, seg_num_x, num_patterns, period_len]
        
        # Attention scores
        scores = torch.matmul(queries, keys.transpose(-1, -2)) / (self.patterns.shape[-1] ** 0.5)
        attention = torch.softmax(scores, dim=-1)  # [B*C, seg_num_x, 1, num_patterns]
        
        # Apply attention
        values = self.value_proj(patterns)  # [B*C, seg_num_x, num_patterns, period_len]
        seasonal_patterns = torch.matmul(attention, values).squeeze(2)  # [B*C, seg_num_x, period_len]
        
        # Kết hợp với dữ liệu gốc
        return x + seasonal_patterns

class SegmentInteraction(nn.Module):
    def __init__(self, seq_len, period_len):
        super().__init__()
        self.segment_ffn = nn.Sequential(
            nn.Linear(period_len, period_len*2),
            nn.GELU(),
            nn.Linear(period_len*2, period_len)
        )
        self.norm = nn.LayerNorm(period_len)
        
    def forward(self, x):
        # x: [B*C, seg_num_x, period_len]
        return x + self.norm(self.segment_ffn(x))

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
        
        # SeasonalPatternAttention
        self.seasonal_pattern_attn = SeasonalPatternAttention(
            period_len=self.period_len,
            num_patterns=8  # Số lượng mẫu mùa vụ cần học
        )
        
        # Thay thế MultiheadAttention bằng SegmentInteraction
        self.segment_interaction = SegmentInteraction(
            seq_len=seq_len,
            period_len=period_len
        )
        
        # Normalization layer
        self.norm1 = nn.LayerNorm(self.period_len)
        
        # MLP cải tiến
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

        # Reshape directly without using conv1d and pool
        s = s.reshape(-1, self.seg_num_x, self.period_len)  # [B*C, seg_num_x, period_len]
        
        # Apply SeasonalPatternAttention
        s = self.seasonal_pattern_attn(s)
        
        # Thêm thông tin thời gian nếu có
        if seq_x_mark is not None:
            # Xử lý time features
            time_embed = self.time_embedding(seq_x_mark)  # [B, Input, d_model]
            
            # Reshape để phù hợp với dữ liệu phân đoạn
            time_embed = time_embed.reshape(B, self.seg_num_x, -1)
            
            # Mở rộng cho mỗi kênh và cắt để khớp period_len
            time_embed = time_embed.unsqueeze(1).expand(-1, C, -1, -1)  # [B, C, seg_num_x, d_model]
            time_embed = time_embed.reshape(B*C, self.seg_num_x, -1)  # [B*C, seg_num_x, d_model]
            
            # Inject time information
            s = s + time_embed[:, :, :self.period_len]
        
        # Thêm positional encoding
        s = self.pos_encoder(s)
        
        # Segment interaction (thay thế cho MultiheadAttention)
        s = self.segment_interaction(s)
        
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