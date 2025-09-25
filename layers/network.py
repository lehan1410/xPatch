import torch
import torch.nn as nn
import numpy as np

class AnchorInterpolationModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        
        # Parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.model_type = configs.model_type
        
        # Anchor parameters
        self.num_anchors = configs.pred_len // configs.period_len  # Số lượng anchor points
        
        # Calculate segments
        self.seg_num_x = self.seq_len // self.period_len
        
        # Sliding aggregation (giảm nhiễu đầu vào)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, 
                               kernel_size=1 + 2 * (self.period_len // 2),
                               stride=1, padding=self.period_len // 2, 
                               padding_mode="zeros", bias=False)
        
        # Anchor predictor - Mô hình dự báo các điểm anchor đơn giản
        if self.model_type == 'linear':
            self.anchor_predictor = nn.Linear(self.seg_num_x, self.num_anchors, bias=False)
        elif self.model_type == 'mlp':
            self.anchor_predictor = nn.Sequential(
                nn.Linear(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.num_anchors)
            )

        # Nội suy - tất cả các lớp cần thiết được triển khai trực tiếp
        # Sử dụng code từ Network nhưng được tích hợp trực tiếp
        
        # Tham số cho nội suy
        interp_seq_len = self.num_anchors  # Input là số lượng anchor
        patch_len = min(4, self.num_anchors//2)
        stride = 1
        padding_patch = 'end'
        
        # Non-linear Stream
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (interp_seq_len - patch_len)//stride + 1
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1
            
        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)
        
        # CNN Depthwise
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)
        
        # Residual Stream
        self.fc2 = nn.Linear(self.dim, patch_len)
        
        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)
        
        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, self.pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(self.pred_len * 2, self.pred_len)
        
        # Linear Stream - MLP
        self.fc5 = nn.Linear(interp_seq_len, self.pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(self.pred_len * 2)
        
        self.fc6 = nn.Linear(self.pred_len * 2, self.pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(self.pred_len // 2)
        
        self.fc7 = nn.Linear(self.pred_len // 2, self.pred_len)
        
        # Streams Concatination
        self.fc8 = nn.Linear(self.pred_len * 2, self.pred_len)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. Normalization và chuyển vị
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)  # [batch, channel, seq_len]
        
        # 2. Sliding aggregation (Conv1D)
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x
        
        # 3. Downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        
        # 4. Dự đoán anchor points
        if self.model_type == 'linear':
            anchor_values = self.anchor_predictor(x.mean(dim=1))  # bc,m
        elif self.model_type == 'mlp':
            anchor_values = self.anchor_predictor(x.mean(dim=1))  # bc,m
        
        # Reshape anchor values về dạng [batch, channel, anchor_points]
        anchor_values = anchor_values.view(batch_size, self.enc_in, self.num_anchors)
        
        # 5. Nội suy các điểm còn lại - dùng logic từ Network
        # Tách thành 2 streams giống Network
        s = anchor_values  # Seasonality
        t = anchor_values  # Trend - có thể dùng anchor hoặc tạo trend khác
        
        # Nội suy trực tiếp - Copy phần forward từ Network
        s = s.permute(0,2,1)  # [batch, anchor_points, channel]
        t = t.permute(0,2,1)  # [batch, anchor_points, channel]
        
        # Channel split
        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        s = torch.reshape(s, (B*C, I))
        t = torch.reshape(t, (B*C, I))
        
        # Non-linear Stream
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # Patch Embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)
        
        res = s
        
        # CNN Depthwise
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)
        
        # Residual Stream
        res = self.fc2(res)
        s = s + res
        
        # CNN Pointwise
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)
        
        # Flatten Head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)
        
        # Linear Stream
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)
        
        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)
        
        t = self.fc7(t)
        
        # Streams Concatination
        x = torch.cat((s, t), dim=1)
        x = self.fc8(x)
        
        # Channel concatination
        x = torch.reshape(x, (B, C, self.pred_len))
        
        # Permute & denormalize
        x = x.permute(0,2,1) + seq_mean
        
        return x