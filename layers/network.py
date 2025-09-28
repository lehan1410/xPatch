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

        # Multi-Scale Convolution cho Seasonal Stream
        self.conv1d_1 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv1d_2 = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        self.conv1d_3 = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        self.conv1d_4 = nn.Conv1d(1, 1, kernel_size=9, padding=4, bias=False)
        self.pool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        
        # Feature fusion cho multi-scale
        self.feature_fusion = nn.Conv1d(5, 1, kernel_size=1, bias=False)  # 4 conv + 1 pool = 5 channels

        # ResidualMLP cho segment processing
        self.mlp = ResidualMLP(self.seg_num_x, self.d_model, self.seg_num_y)

        # Improved Linear Stream với residual connections
        self.fc5 = nn.Linear(seq_len, pred_len * 2)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.dropout1 = nn.Dropout(0.1)
        
        # Thêm intermediate layer với skip connection
        self.fc6 = nn.Linear(pred_len * 2, pred_len * 2)
        self.gelu2 = nn.GELU()
        self.ln2 = nn.LayerNorm(pred_len * 2)
        self.dropout2 = nn.Dropout(0.1)
        
        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        self.fc8 = nn.Linear(pred_len, pred_len)

        # Enhanced Cross-stream với learnable weights
        self.cross_weight = nn.Parameter(torch.tensor(0.1))
        self.cross_connection_t = nn.Linear(self.pred_len, self.pred_len)
        self.cross_connection_y = nn.Linear(self.pred_len, self.pred_len)

    def forward(self, s, t):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        s = s.permute(0,2,1) # [Batch, Channel, Input]
        t = t.permute(0,2,1) # [Batch, Channel, Input]

        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        t = torch.reshape(t, (B*C, I))

        # Multi-Scale Seasonal Stream
        s_input = s.reshape(-1, 1, self.seq_len)
        s_conv1 = self.conv1d_1(s_input)
        s_conv2 = self.conv1d_2(s_input)
        s_conv3 = self.conv1d_3(s_input)
        s_conv4 = self.conv1d_4(s_input)
        s_pool = self.pool(s_input)
        
        # Concatenate và fuse multi-scale features
        s_multi = torch.cat([s_conv1, s_conv2, s_conv3, s_conv4, s_pool], dim=1)
        s_fused = self.feature_fusion(s_multi)  # [B*C, 1, seq_len]
        
        # Residual connection với original input
        s_concat = s_fused.reshape(-1, self.enc_in, self.seq_len) + s
        
        # Reshape để đưa vào MLP
        s_reshaped = s_concat.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        y = self.mlp(s_reshaped)
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1)  # [B, pred_len, enc_in]

        # Improved Linear Stream với skip connections
        t_orig = t.clone()
        
        # First transformation
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.dropout1(t)
        
        # Intermediate layer với residual connection
        t_res = t.clone()
        t = self.fc6(t)
        t = self.gelu2(t)
        t = self.ln2(t)
        t = self.dropout2(t)
        t = t + t_res  # Skip connection
        
        # Final layers
        t = self.fc7(t)
        t = self.fc8(t)
        
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1)

        # Enhanced Cross-stream với learnable weights
        t_enhanced = self.cross_connection_t(t.transpose(-2, -1)).transpose(-2, -1)
        y_enhanced = self.cross_connection_y(y.transpose(-2, -1)).transpose(-2, -1)
        
        # Adaptive combination với learnable weight
        cross_weight = torch.sigmoid(self.cross_weight)  # Ensure 0-1 range
        return (t + cross_weight * y_enhanced) + (y + cross_weight * t_enhanced)