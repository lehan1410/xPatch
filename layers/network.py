import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, c_in, period_len, d_model):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in  = c_in
        self.period_len = 24
        self.d_model = 128

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

        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.seg_num_y)
        )

        # Improved Linear Stream với residual connections
        self.fc5 = nn.Linear(seq_len, pred_len * 2)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.dropout1 = nn.Dropout(0.1)
        
        # Thêm intermediate layers với residual connection
        self.fc6 = nn.Linear(pred_len * 2, pred_len * 2)
        self.gelu2 = nn.GELU()
        self.ln2 = nn.LayerNorm(pred_len * 2)
        self.dropout2 = nn.Dropout(0.1)
        
        # Expansion layer để tăng capacity
        self.fc_expand = nn.Linear(pred_len * 2, pred_len * 4)
        self.gelu3 = nn.GELU()
        self.ln3 = nn.LayerNorm(pred_len * 4)
        self.dropout3 = nn.Dropout(0.15)
        
        # Compression layers
        self.fc7 = nn.Linear(pred_len * 4, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.ln4 = nn.LayerNorm(pred_len * 2)
        
        self.fc8 = nn.Linear(pred_len * 2, pred_len)
        self.fc9 = nn.Linear(pred_len, pred_len)  # Final refinement layer
        
        # Skip connection từ đầu vào (nếu cần)
        self.use_input_skip = (seq_len == pred_len)
        if not self.use_input_skip:
            self.input_projection = nn.Linear(seq_len, pred_len)

    def forward(self, s, t):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        s = s.permute(0,2,1) # [Batch, Channel, Input]
        t = t.permute(0,2,1) # [Batch, Channel, Input]

        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        t = torch.reshape(t, (B*C, I))
        
        # Lưu input gốc cho skip connection
        t_input = t.clone()

        # Seasonal Stream: Conv1d + Pooling
        s_conv = self.conv1d(s.reshape(-1, 1, self.seq_len))
        s_pool = self.pool(s.reshape(-1, 1, self.seq_len))
        s_concat = s_conv + s_pool
        s_concat = s_concat.reshape(-1, self.enc_in, self.seq_len) + s
        s = s_concat.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        y = self.mlp(s)
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1) # [B, pred_len, enc_in]

        # Improved Linear Stream với multiple residual connections
        # Block 1: Initial transformation
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.dropout1(t)
        
        # Block 2: Intermediate processing với residual
        t_res1 = t.clone()
        t = self.fc6(t)
        t = self.gelu2(t)
        t = self.ln2(t)
        t = self.dropout2(t)
        t = t + t_res1  # First residual connection
        
        # Block 3: Expansion cho increased capacity
        t = self.fc_expand(t)
        t = self.gelu3(t)
        t = self.ln3(t)
        t = self.dropout3(t)
        
        # Block 4: Compression với residual
        t_expanded = t.clone()
        t = self.fc7(t)
        t = self.gelu4(t)
        t = self.ln4(t)
        # Residual connection (cần project expanded features)
        t_proj = torch.mean(t_expanded.view(*t_expanded.shape[:-1], 2, -1), dim=-2)  # Average pooling để giảm dim
        t = t + t_proj
        
        # Block 5: Final layers
        t = self.fc8(t)
        t_final_res = t.clone()
        t = self.fc9(t)
        t = t + t_final_res  # Final residual
        
        # Skip connection từ input (nếu có thể)
        if self.use_input_skip:
            t = t + t_input
        else:
            t_input_proj = self.input_projection(t_input)
            t = t + t_input_proj
        
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [Batch, Output, Channel] = [B, pred_len, C]

        return t + y