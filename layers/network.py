import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, period_len=24, enc_in=1):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.period_len = period_len
        self.d_model = 108

        # Non-linear Stream
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        
        self.dim = patch_len
        self.patch_num = (seq_len - patch_len)//stride + 1
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            self.patch_num += 1
        
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1 + 2 * (self.period_len // 2),
                                stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False)

        # self.mlp = nn.Sequential(
        #         nn.Linear(self.seg_num_x, self.d_model),
        #         nn.ReLU(),
        #         nn.Linear(self.d_model, self.seg_num_y)
        #     )

        # # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * self.dim, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)
        
        # # CNN Depthwise
        # self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
        #                        patch_len, patch_len, groups=self.patch_num)
        # self.gelu2 = nn.GELU()
        # self.bn2 = nn.BatchNorm1d(self.patch_num)

        # # Residual Stream
        # self.fc2 = nn.Linear(self.dim, patch_len)

        # # CNN Pointwise
        # self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        # self.gelu3 = nn.GELU()
        # self.bn3 = nn.BatchNorm1d(self.patch_num)

        # # Flatten Head
        # self.flatten1 = nn.Flatten(start_dim=-2)
        # self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        # self.gelu4 = nn.GELU()
        # self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # # Linear Stream
        # # MLP
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # Streams Concatination
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s, t):
        # x: [Batch, Input, Channel]
        # s - seasonality
        # t - trend

        batch_size = s.shape[0]
        seq_mean = torch.mean(s, dim=1).unsqueeze(1)  # [B, 1, C]
        s = (s - seq_mean).permute(0, 2, 1)           # [B, C, seq_len]
        B, C, S = s.shape

        # Channel split for channel independence
        s = torch.reshape(s, (B*C, S)) # [B*C, seq_len]

        # Patching
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        patches = s.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [B*C, patch_num, patch_len]

        # Downsample từng patch
        patches = patches.unsqueeze(2) # [B*C, patch_num, 1, patch_len]
        patches = patches.reshape(-1, 1, self.patch_len) # [B*C*patch_num, 1, patch_len]
        patches = self.patch_downsample(patches) # [B*C*patch_num, 1, patch_len//2]
        patches = patches.squeeze(1)   # [B*C*patch_num, patch_len//2]

        # Patch embedding
        patches = self.fc1(patches)    # [B*C*patch_num, dim]
        patches = self.gelu1(patches)
        patches = patches.reshape(B*C, self.patch_num, self.dim)
        patches = self.bn1(patches)

        # Flatten Head
        patches = self.flatten1(patches) # [B*C, patch_num * dim]
        patches = self.fc3(patches)
        patches = self.gelu4(patches)
        s = self.fc4(patches)            # [B*C, pred_len]

        # Linear Stream (trend)
        t = t.permute(0,2,1)                               # [B, C, seq_len]
        t = torch.reshape(t, (batch_size*self.enc_in, self.seq_len))
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
        x = torch.reshape(x, (B, C, self.pred_len)) # [Batch, Channel, Output]
        x = x.permute(0,2,1) # to [Batch, Output, Channel]

        return x