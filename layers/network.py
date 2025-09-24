import torch
from torch import nn

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
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        # GLU removed here to keep deterministic dims; final fc4 matches GLU-aware shape
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # Linear Stream (trend / simple path)
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # --- New: simple baseline and gating to adapt complexity ---
        # simple linear baseline per channel (fast, good for simple datasets)
        self.simple_baseline = nn.Linear(seq_len, pred_len)
        # gate that learns how to combine complex vs baseline (per output step)
        self.gate_fc = nn.Linear(pred_len * 2, pred_len)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)
        # ----------------------------------------------------------------

        # Streams Concatination (complex s+t path -> pred_len, then will be gated)
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s, t, c=None, r=None):
        # x: [Batch, Input, Channel]
        # s - seasonality
        # t - trend

        if c is not None:
            s = s + c
        # If irregular provided, merge into trend/linear stream as residual
        if r is not None:
            t = t + r
        
        s = s.permute(0,2,1) # to [Batch, Channel, Input]
        t = t.permute(0,2,1) # to [Batch, Channel, Input]
        
        # Channel split for channel independence
        B = s.shape[0] # Batch size
        C = s.shape[1] # Channel size
        I = s.shape[2] # Input size
        s = torch.reshape(s, (B*C, I)) # [Batch and Channel, Input]
        t = torch.reshape(t, (B*C, I)) # [Batch and Channel, Input]

        # Keep copy of raw input for simple baseline
        t_raw = t  # [N, seq_len]

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
        s = self.fc4(s)               # complex seasonal output part -> shape [N, pred_len]

        # Linear Stream (complex trend processing)
        t_cpx = self.fc5(t)
        t_cpx = self.avgpool1(t_cpx)
        t_cpx = self.ln1(t_cpx)

        t_cpx = self.fc6(t_cpx)
        t_cpx = self.avgpool2(t_cpx)
        t_cpx = self.ln2(t_cpx)

        t_cpx = self.fc7(t_cpx)       # shape [N, pred_len]

        # Streams Concatination (complex)
        complex_out = torch.cat((s, t_cpx), dim=1)   # [N, pred_len*2]
        complex_out = self.dropout(complex_out)
        complex_out_proj = self.fc8(complex_out)     # [N, pred_len]

        # Simple baseline (fast linear mapping)
        baseline = self.simple_baseline(t_raw)       # [N, pred_len]
        baseline = self.dropout(baseline)

        # Gate: decide per-output linear combination between complex and baseline
        gate_input = torch.cat((complex_out_proj, baseline), dim=1)  # [N, pred_len*2]
        gate = self.sigmoid(self.gate_fc(gate_input))                # [N, pred_len]

        # Final adaptive blend (allows model to fallback to baseline for simple datasets)
        out = gate * complex_out_proj + (1.0 - gate) * baseline     # [N, pred_len]

        # Reshape back
        x = out.reshape(B, C, self.pred_len) # [Batch, Channel, Output]
        x = x.permute(0,2,1) # to [Batch, Output, Channel]

        return x