import torch
from torch import nn

class HighwayConnection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transform = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        
    def forward(self, x):
        H = self.transform(x)
        T = torch.sigmoid(self.gate(x))
        return H * T + x * (1 - T)

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len

        # Non-linear Stream
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len)//stride + 1
        if padding_patch == 'end': # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            self.patch_num += 1

        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num, patch_len, patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)
        self.fc2 = nn.Linear(self.dim, patch_len)
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # Highway connections for each component
        self.highway_s = HighwayConnection(pred_len)
        self.highway_c = HighwayConnection(pred_len)
        self.highway_r = HighwayConnection(pred_len)
        self.highway_t = HighwayConnection(pred_len)

        # Linear Stream
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)
        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        self.head_s = nn.Linear(pred_len, pred_len)
        self.head_c = nn.Linear(pred_len, pred_len)
        self.head_r = nn.Linear(pred_len, pred_len)
        self.head_t = nn.Linear(pred_len, pred_len)

        # Streams Concatination
        self.fc8 = nn.Linear(pred_len * 4, pred_len)

    def forward(self, s, t, c=None, r=None):
        # x: [Batch, Input, Channel]
        # s - seasonality
        # t - trend

        s = s.permute(0,2,1)
        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        s = torch.reshape(s, (B*C, I))

        # Non-linear Stream
        # Patching
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: [Batch and Channel, Patch_num, Patch_len]
        
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)
        res_s = s
        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)
        res_s = self.fc2(res_s)
        s = s + res_s
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)
        s = self.head_s(s)
        s = self.highway_s(s)

        if c is not None:
            c = c.permute(0,2,1)
            c = torch.reshape(c, (B*C, I))
            if self.padding_patch == 'end':
                c = self.padding_patch_layer(c)
            c = c.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            c = self.fc1(c)
            c = self.gelu1(c)
            c = self.bn1(c)
            res_c = c
            c = self.conv1(c)
            c = self.gelu2(c)
            c = self.bn2(c)
            res_c = self.fc2(res_c)
            c = c + res_c
            c = self.conv2(c)
            c = self.gelu3(c)
            c = self.bn3(c)
            c = self.flatten1(c)
            c = self.fc3(c)
            c = self.gelu4(c)
            c = self.fc4(c)
            c = self.head_c(c)
            c = self.highway_c(c)
        else:
            c = torch.zeros_like(s)

        if r is not None:
            r = r.permute(0,2,1)
            r = torch.reshape(r, (B*C, I))
            if self.padding_patch == 'end':
                r = self.padding_patch_layer(r)
            r = r.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            r = self.fc1(r)
            r = self.gelu1(r)
            r = self.bn1(r)
            res_r = r
            r = self.conv1(r)
            r = self.gelu2(r)
            r = self.bn2(r)
            res_r = self.fc2(res_r)
            r = r + res_r
            r = self.conv2(r)
            r = self.gelu3(r)
            r = self.bn3(r)
            r = self.flatten1(r)
            r = self.fc3(r)
            r = self.gelu4(r)
            r = self.fc4(r)
            r = self.head_r(r)
            r = self.highway_r(r)
        else:
            r = torch.zeros_like(s)

        # Linear Stream
        # MLP
        t = t.permute(0,2,1)
        t = torch.reshape(t, (B*C, I))
        t = self.fc5(t)
        t = self.avgpool1(t.unsqueeze(1)).squeeze(1)
        t = self.ln1(t)
        t = self.fc6(t)
        t = self.avgpool2(t.unsqueeze(1)).squeeze(1)
        t = self.ln2(t)
        t = self.fc7(t)
        t = self.head_t(t)
        t = self.highway_t(t)

        # Streams Concatination
        x = torch.cat((s, t, c, r), dim=1)
        x = self.fc8(x)

        # Channel concatination
        x = x.reshape(B, C, self.pred_len) # [Batch, Channel, Output]

        x = x.permute(0,2,1) # to [Batch, Output, Channel]

        return x