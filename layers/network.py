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
        # final seasonal complex head (compatible with GLU removal)
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # Linear Stream (trend / complex path)
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # --- Baselines & gates to adapt complexity (avoid harming simple datasets) ---
        # simple linear baseline for trend (t)
        self.simple_baseline = nn.Linear(seq_len, pred_len)
        # small linear baselines for cyclic and irregular when provided
        self.simple_baseline_c = nn.Linear(seq_len, pred_len)
        self.simple_baseline_r = nn.Linear(seq_len, pred_len)

        # gate that learns how to combine complex vs baseline (per output step)
        self.gate_fc = nn.Linear(pred_len * 2, pred_len)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)

        # learnable residual weights for c,r contributions (per output)
        self.alpha_c = nn.Parameter(torch.zeros(pred_len))
        self.alpha_r = nn.Parameter(torch.zeros(pred_len))
        # ----------------------------------------------------------------

        # Streams Concatination (complex s+t path -> pred_len, then will be gated)
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s, t, c=None, r=None):
        # x: [Batch, Input, Channel]
        # s - seasonality, t - trend
        # c, r optional — NOT added directly into s/t

        # keep originals for separate baseline processing
        c_in = c
        r_in = r

        s = s.permute(0,2,1) # to [Batch, Channel, Input]
        t = t.permute(0,2,1) # to [Batch, Channel, Input]
        
        # Channel split for channel independence
        B = s.shape[0] # Batch size
        C = s.shape[1] # Channel size
        I = s.shape[2] # Input size
        s = torch.reshape(s, (B*C, I)) # [N, seq_len]
        t = torch.reshape(t, (B*C, I)) # [N, seq_len]

        # Keep copy of raw input for simple baseline (trend)
        t_raw = t  # [N, seq_len]
        baseline_t = self.simple_baseline(t_raw)   # [N, pred_len]

        # Baselines for c and r (if provided)
        if c_in is not None:
            c_tmp = c_in.permute(0,2,1).reshape(B*C, I)
            baseline_c = self.simple_baseline_c(c_tmp)
        else:
            baseline_c = torch.zeros_like(baseline_t, device=baseline_t.device)

        if r_in is not None:
            r_tmp = r_in.permute(0,2,1).reshape(B*C, I)
            baseline_r = self.simple_baseline_r(r_tmp)
        else:
            baseline_r = torch.zeros_like(baseline_t, device=baseline_t.device)

        # ----------------- complex non-linear seasonal stream -----------------
        # Patching
        if self.padding_patch == 'end':
            s_pad = self.padding_patch_layer(s)
        else:
            s_pad = s
        s_unf = s_pad.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s_unf: [N, Patch_num, Patch_len]

        # Patch Embedding
        s_e = self.fc1(s_unf)
        s_e = self.gelu1(s_e)
        s_e = self.bn1(s_e)

        res = s_e

        # CNN Depthwise
        s_e = self.conv1(s_e)
        s_e = self.gelu2(s_e)
        s_e = self.bn2(s_e)

        # Residual Stream
        res = self.fc2(res)
        s_e = s_e + res

        # CNN Pointwise
        s_e = self.conv2(s_e)
        s_e = self.gelu3(s_e)
        s_e = self.bn3(s_e)

        # Flatten Head -> complex seasonal part
        s_flat = self.flatten1(s_e)
        s_flat = self.fc3(s_flat)
        s_flat = self.gelu4(s_flat)
        s_complex = self.fc4(s_flat)               # [N, pred_len]

        # ----------------- complex trend stream -----------------
        t_cpx = self.fc5(t)
        # avgpool1 expects [N, C, L], but here fc5 returns [N, pred_len*4]; keep compatible:
        # use unsqueeze/avgpool then squeeze (same as original design)
        t_cpx = self.avgpool1(t_cpx.unsqueeze(1)).squeeze(1)
        t_cpx = self.ln1(t_cpx)

        t_cpx = self.fc6(t_cpx)
        t_cpx = self.avgpool2(t_cpx.unsqueeze(1)).squeeze(1)
        t_cpx = self.ln2(t_cpx)

        t_cpx = self.fc7(t_cpx)       # shape [N, pred_len]

        # Streams Concatination (complex)
        complex_out = torch.cat((s_complex, t_cpx), dim=1)   # [N, pred_len*2]
        complex_out = self.dropout(complex_out)
        complex_out_proj = self.fc8(complex_out)     # [N, pred_len]

        # Adaptive gating between complex path and baseline trend
        gate_input = torch.cat((complex_out_proj, baseline_t), dim=1)  # [N, pred_len*2]
        gate = self.sigmoid(self.gate_fc(gate_input))                  # [N, pred_len]
        out = gate * complex_out_proj + (1.0 - gate) * baseline_t      # [N, pred_len]

        # Add contributions from c and r as learnable residuals (if present)
        alpha_c = torch.sigmoid(self.alpha_c).unsqueeze(0)  # [1, pred_len]
        alpha_r = torch.sigmoid(self.alpha_r).unsqueeze(0)
        out = out + alpha_c * baseline_c + alpha_r * baseline_r

        # Reshape back
        x = out.reshape(B, C, self.pred_len) # [Batch, Channel, Output]
        x = x.permute(0,2,1) # to [Batch, Output, Channel]

        return x