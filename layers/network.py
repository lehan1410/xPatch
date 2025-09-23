import torch
from torch import nn
import torch.nn.functional as F

class Shrinkage(nn.Module):
    """Learnable soft-threshold shrinkage on [..., T, C] with per-channel thresholds."""
    def __init__(self, n_channels):
        super().__init__()
        # threshold per channel (init small)
        self.theta = nn.Parameter(torch.full((1, 1, n_channels), 1e-2))

    def forward(self, x):
        # x: [B, T, C]
        theta = torch.relu(self.theta)  # ensure non-negative
        sign = torch.sign(x)
        out = sign * F.relu(torch.abs(x) - theta)
        return out

def trend_smoothness_loss(trend):
    # trend: [B, T, C] -> second-order finite difference along time
    d2 = trend[:, 2:, :] - 2 * trend[:, 1:-1, :] + trend[:, :-2, :]
    return (d2 ** 2).mean()

class DilatedDepthwiseBlock(nn.Module):
    """
    Multi-scale dilated depthwise block.
    Applies several depthwise Conv1d layers with exponential dilation (1,2,4,...),
    batchnorm  SiLU, and residual gating.
    Input shape: (batch, channels=patch_num, length=dim)
    """
    def __init__(self, channels, kernel_size, layers=3, activation='silu'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.channels = channels
        self.activation = activation
        for i in range(layers):
            dilation = 2 ** i
            pad = dilation * (kernel_size - 1) // 2
            conv = nn.Conv1d(channels, channels, kernel_size,
                             padding=pad, dilation=dilation, groups=channels, bias=False)
            bn = nn.BatchNorm1d(channels)
            act = nn.SiLU() if activation == 'silu' else nn.Mish()
            self.layers.append(nn.Sequential(conv, bn, act))
        # small gate for residual mixing (per-channel scalar)
        self.gate_fc = nn.Linear(channels, channels)
        nn.init.constant_(self.gate_fc.bias, -3.0)  # favor identity initially

    def forward(self, x):
        # x: [B, C, L]
        out = x
        for layer in self.layers:
            out = layer(out)

        # ensure out length matches input by center-cropping or padding if needed
        in_len = x.size(2)
        out_len = out.size(2)
        if out_len != in_len:
            if out_len > in_len:
                start = (out_len - in_len) // 2
                out = out[:, :, start:start + in_len]
            else:
                pad_total = in_len - out_len
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                out = F.pad(out, (pad_left, pad_right))  # pad along time dim

        # compute per-channel gate from global context
        ctx = out.mean(dim=2)                 # [B, C]
        g = torch.sigmoid(self.gate_fc(ctx)).unsqueeze(-1)  # [B, C, 1]
        # gated residual: mix transformed and input
        return g * out + (1.0 - g) * x

class PointwiseGLU(nn.Module):
    """1x1 pointwise with GLU activation for stronger gating nonlinearity."""
    def __init__(self, channels):
        super().__init__()
        # produce 2x channels then GLU along channel dim -> returns channels
        self.conv = nn.Conv1d(channels, channels * 2, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        # x: [B, C, L]
        x = self.conv(x)               # [B, 2C, L]
        x = nn.functional.glu(x, dim=1)  # [B, C, L]
        x = self.bn(x)
        return x

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

        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)
        
        # CNN Depthwise
        # self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
        #                        patch_len, patch_len, groups=self.patch_num)
        # self.gelu2 = nn.GELU()
        # self.bn2 = nn.BatchNorm1d(self.patch_num)
        self.dilated_block = DilatedDepthwiseBlock(self.patch_num, kernel_size=patch_len, layers=3, activation='silu')

        # Residual Stream
        self.fc2 = nn.Linear(self.dim, self.dim)

        # CNN Pointwise
        # self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        # self.gelu3 = nn.GELU()
        # self.bn3 = nn.BatchNorm1d(self.patch_num)
        self.pointwise_glu = PointwiseGLU(self.patch_num)
        self.gelu3 = nn.GELU()

        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * self.dim, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # Linear Stream
        # MLP
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        self.noise_net = nn.Sequential(
            nn.Linear(pred_len * 2, pred_len),
            nn.ReLU(),
            nn.Linear(pred_len, pred_len)
        )

        self._shrinkage = None

        self.gate_fc1 = nn.Linear(pred_len, pred_len//2)
        self.gate_act = nn.ReLU()
        self.gate_fc2 = None

        # Streams Concatination
        self.fc8 = nn.Linear(pred_len * 2, pred_len)
    
    def _init_channel_modules(self, n_channels, device):
        # create shrinkage and gate_fc2 with correct channel size
        if self._shrinkage is None:
            self._shrinkage = Shrinkage(n_channels).to(device)
        if self.gate_fc2 is None:
            self.gate_fc2 = nn.Linear(self.gate_fc1.out_features, n_channels).to(device)
            # bias init negative to favor trend initially
            with torch.no_grad():
                if self.gate_fc2.bias is not None:
                    self.gate_fc2.bias.fill_(-3.0)

    def forward(self, s, t):
        # x: [Batch, Input, Channel]
        # s - seasonality
        # t - trend
        
        s = s.permute(0,2,1) # to [Batch, Channel, Input]
        t = t.permute(0,2,1) # to [Batch, Channel, Input]
        
        # Channel split for channel independence
        B = s.shape[0] # Batch size
        C = s.shape[1] # Channel size
        I = s.shape[2] # Input size
        s = torch.reshape(s, (B*C, I)) # [Batch and Channel, Input]
        t = torch.reshape(t, (B*C, I)) # [Batch and Channel, Input]

        # Non-linear Stream
        # Patching
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: [Batch and Channel, Patch_num, Patch_len]
        
        # Patch Embedding
        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        # CNN Depthwise
        # s = self.conv1(s)
        # s = self.gelu2(s)
        # s = self.bn2(s)
        s = self.dilated_block(s)

        # Residual Stream
        res = self.fc2(res)
        s = s + res

        # CNN Pointwise
        # s = self.conv2(s)
        # s = self.gelu3(s)
        # s = self.bn3(s)
        s = self.pointwise_glu(s)
        s = self.gelu3(s)

        # Flatten Head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        # Linear Stream
        # MLP
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)

        combined_feat = torch.cat((s, t), dim=1)  # [B*C, pred_len*2]
        noise_raw = self.noise_net(combined_feat)
        noise = noise_raw.view(B, C, self.pred_len).permute(0,2,1)
        noise = self._shrinkage(noise)   # shrink sparse noise
        # reshape back to [B*C, pred_len]
        noise_pred = noise.permute(0,2,1).contiguous().view(B*C, self.pred_len)

        trend_for_gate = t.view(B, C, self.pred_len).permute(0,2,1)  # [B, pred_len, C]
        ctx = trend_for_gate.mean(dim=1)  # [B, C]
        g_h = self.gate_act(self.gate_fc1(ctx))  # [B, hidden]
        g_logits = self.gate_fc2(g_h)             # [B, C]
        gates = torch.sigmoid(g_logits).unsqueeze(1)

        seasonal_out = s.view(B, C, self.pred_len).permute(0,2,1)
        trend_out = t.view(B, C, self.pred_len).permute(0,2,1)
        learned = seasonal_out + noise
        out = (1.0 - gates) * trend_out + gates * learned

        return out