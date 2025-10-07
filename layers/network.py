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
        self.cycle_len = period_len  # Sử dụng period_len làm cycle_len
        
        # Thêm tham số phân đoạn
        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        # Temporal Query - học các mẫu theo chu kỳ
        self.temporalQuery = torch.nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True)

        # Channel Aggregator - tích hợp thông tin giữa các kênh
        self.channelAggregator = nn.MultiheadAttention(
            embed_dim=self.seq_len, 
            num_heads=4, 
            batch_first=True, 
            dropout=0.1
        )

        # Thêm Conv1D trước khi dự đoán
        self.conv1d = nn.Conv1d(
            in_channels=self.enc_in, out_channels=self.enc_in,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False, groups=self.enc_in
        )

        # MLP cho dự đoán phân đoạn - như trong net1.py
        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.seg_num_y)
        )

        # Linear Stream (giữ nguyên)
        self.fc5 = nn.Linear(seq_len, pred_len * 2)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t, cycle_index):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        s = s.permute(0, 2, 1)  # [Batch, Channel, Input]
        t = t.permute(0, 2, 1)  # [Batch, Channel, Input]

        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        t = torch.reshape(t, (B*C, I))

        # Seasonal Stream (với cycle-based attention)
        # 1. Tạo temporal query dựa trên cycle_index
        gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
        query_input = self.temporalQuery[gather_index].permute(0, 2, 1)  # (B, C, seq_len)
        
        # 2. Channel aggregation
        channel_information = self.channelAggregator(query=query_input, key=s, value=s)[0]
        
        # 3. Kết hợp thông tin
        s_combined = s + channel_information
        
        # 4. Áp dụng Conv1D trước khi dự đoán
        s_conv = self.conv1d(s_combined)  # [B, C, seq_len]
        s_combined = s_conv + s_combined  # Thêm residual connection
        
        # 5. Xử lý giống như trong net1.py
        s_combined = s_combined.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)  # [B*C, period_len, seg_num_x]
        y = self.mlp(s_combined)  # [B*C, period_len, seg_num_y]
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)  # [B, C, pred_len]
        y = y.permute(0, 2, 1)  # [B, pred_len, C]

        # Linear Stream (giữ nguyên)
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0, 2, 1)  # [Batch, Output, Channel] = [B, pred_len, C]

        return t + y