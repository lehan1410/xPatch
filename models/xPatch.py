import torch
import torch.nn as nn
from layers.revin import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = 24
        self.d_model = 108
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.padding_patch = configs.padding_patch

        self.patch_num = (self.seq_len - self.patch_len)//self.stride + 1
        if self.padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
            self.patch_num += 1

        # RevIN layer
        self.revin_layer = RevIN(self.enc_in, affine=True, subtract_last=False)

        # Patch Embedding
        self.fc1 = nn.Linear(self.patch_len, self.d_model)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # CNN Depthwise
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num, self.patch_len, self.patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual Stream
        self.fc2 = nn.Linear(self.d_model, self.patch_len)

        # CNN Pointwise
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * self.patch_len, self.seq_len)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(self.seq_len, self.seq_len)

        # Downsampling (aggregation theo period_len)
        self.conv1d = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.seq_len // self.period_len, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.pred_len // self.period_len)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        # RevIN normalization
        x = self.revin_layer(x, 'norm')

        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)  # [B, C, S]
        B, C, S = x.shape
        x = torch.reshape(x, (B*C, S))       # [B*C, S]

        # Patching
        if self.padding_patch == 'end':
            x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [B*C, patch_num, patch_len]

        # Patch Embedding + Local Feature Extraction
        x = self.fc1(x)
        x = self.gelu1(x)
        x = self.bn1(x)

        res = x

        # CNN Depthwise
        x = self.conv1(x)
        x = self.gelu2(x)
        x = self.bn2(x)

        # Residual Stream
        res = self.fc2(res)
        x = x + res

        # CNN Pointwise
        x = self.conv2(x)
        x = self.gelu3(x)
        x = self.bn3(x)

        # Flatten Head
        x = self.flatten1(x)
        x = self.fc3(x)
        x = self.gelu4(x)
        x = self.fc4(x)

        # Downsampling theo period_len
        x = x.reshape(-1, 1, self.seq_len)
        x = self.conv1d(x).reshape(-1, self.enc_in, self.seq_len) + x.reshape(-1, self.enc_in, self.seq_len)

        # downsampling: b,c,s -> bc,n,w -> bc,w,n
        seg_num_x = self.seq_len // self.period_len
        seg_num_y = self.pred_len // self.period_len
        x = x.reshape(-1, seg_num_x, self.period_len).permute(0, 2, 1)

        # Qua MLP để dự đoán
        x = self.mlp(x)  # [B*C, period_len, seg_num_y]

        # upsampling: bc,w,n -> bc,n,w -> b,c,s
        x = x.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

        # permute and denorm
        x = x.permute(0, 2, 1) + seq_mean

        # RevIN denormalization
        x = self.revin_layer(x, 'denorm')

        return x