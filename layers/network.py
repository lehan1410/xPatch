# ...existing code...
import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len

        # Patching params
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len)//stride + 1
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        # ---------------- Non-linear (complex) stream ----------------
        # Patch embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        # Depthwise conv (local features per patch index) - base
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               patch_len, patch_len, groups=self.patch_num)
        self.act2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # --- Multi-scale / dilated depthwise convolutions (lightweight) ---
        self.ms_conv_k3 = nn.Conv1d(self.patch_num, self.patch_num, kernel_size=3, padding=1, groups=self.patch_num)
        self.ms_conv_d2 = nn.Conv1d(self.patch_num, self.patch_num, kernel_size=3, dilation=2, padding=2, groups=self.patch_num)
        self.bn_ms = nn.BatchNorm1d(self.patch_num)
        # ------------------------------------------------------------------

        # Residual projection back to patch length
        self.fc2 = nn.Linear(self.dim, patch_len)

        # Pointwise conv to mix patch channels (local mixing along patch_idx)
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.act3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # ---------------- Cross-channel mixing (learn multi-variate deps) ----------------
        # channel_mixer operates across patch-channel axis; small bottleneck to limit params
        cm_hidden = max(16, self.patch_num // 2)
        self.channel_mixer = nn.Sequential(
            nn.Linear(self.patch_num, cm_hidden),
            nn.GELU(),
            nn.Linear(cm_hidden, self.patch_num)
        )
        # ----------------------------------------------------------------------------------

        # Flatten & complex head (seasonal part)
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.act4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)   # produces seasonal complex output

        # ---------------- Linear (trend) stream ----------------
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)  # produces trend complex output

        # ---------------- Adaptive baseline & gates ----------------
        # simple baselines (help on simple datasets)
        self.simple_baseline = nn.Linear(seq_len, pred_len)
        self.simple_baseline_c = nn.Linear(seq_len, pred_len)
        self.simple_baseline_r = nn.Linear(seq_len, pred_len)

        # gate to blend complex vs baseline
        self.gate_fc = nn.Linear(pred_len * 2, pred_len)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.1)

        # learnable residual scales for c and r
        self.alpha_c = nn.Parameter(torch.zeros(pred_len))
        self.alpha_r = nn.Parameter(torch.zeros(pred_len))

        # final projector for complex s+t combined
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s, t, c=None, r=None):
        # s, t, c, r: [B, L, C] (Input length L, Channels C)
        # do NOT add c -> s or r -> t directly; use learned baselines and gating

        # keep originals for baselines
        c_in = c
        r_in = r

        # permute and merge batch/channel for independent processing
        s = s.permute(0,2,1)  # [B, C, L]
        t = t.permute(0,2,1)
        B, C, L = s.shape
        N = B * C
        s = s.reshape(N, L)   # [N, L]
        t = t.reshape(N, L)   # [N, L]

        # ---- Compute baselines ----
        baseline_t = self.simple_baseline(t)  # [N, pred_len]

        if c_in is not None:
            c_tmp = c_in.permute(0,2,1).reshape(N, L)
            baseline_c = self.simple_baseline_c(c_tmp)
        else:
            baseline_c = torch.zeros_like(baseline_t, device=baseline_t.device)

        if r_in is not None:
            r_tmp = r_in.permute(0,2,1).reshape(N, L)
            baseline_r = self.simple_baseline_r(r_tmp)
        else:
            baseline_r = torch.zeros_like(baseline_t, device=baseline_t.device)

        # ---------------- complex seasonal stream ----------------
        if self.padding_patch == 'end':
            s_pad = self.padding_patch_layer(s)
        else:
            s_pad = s
        s_unf = s_pad.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s_unf: [N, patch_num, patch_len]

        s_e = self.fc1(s_unf)         # [N, patch_num, dim]
        s_e = self.act1(s_e)
        s_e = self.bn1(s_e)

        # base depthwise conv
        s_base = self.conv1(s_e)      # [N, patch_num, dim]
        s_base = self.act2(s_base)
        s_base = self.bn2(s_base)

        # multi-scale / dilated depthwise convs (lightweight)
        s_ms1 = self.ms_conv_k3(s_e)  # local small kernel
        s_ms2 = self.ms_conv_d2(s_e)  # dilated kernel for longer context
        s_ms = s_base + s_ms1 + s_ms2
        s_ms = self.bn_ms(s_ms)

        # residual projection and add
        res = self.fc2(s_e)
        s_ms = s_ms + res

        # pointwise mixing (per-patch mixing)
        s_ms = self.conv2(s_ms)
        s_ms = self.act3(s_ms)
        s_ms = self.bn3(s_ms)

        # cross-channel mixing: move patch-channel to last dim, apply mixer per time-step
        # current s_ms: [N, patch_num, dim] -> transpose to [N, dim, patch_num]
        s_mix = s_ms.transpose(1, 2)            # [N, dim, patch_num]
        # apply same MLP to last dim; nn.Linear supports broadcasting over leading dims
        s_mix = self.channel_mixer(s_mix)       # [N, dim, patch_num]
        s_ms = s_mix.transpose(1, 2)            # [N, patch_num, dim]

        # flatten and seasonal complex head
        s_flat = self.flatten1(s_ms)   # [N, patch_num * patch_len]
        s_flat = self.fc3(s_flat)
        s_flat = self.act4(s_flat)
        s_complex = self.fc4(s_flat)  # [N, pred_len]

        # ---------------- complex trend stream ----------------
        t_cpx = self.fc5(t)                     # [N, pred_len*4]
        t_cpx = self.avgpool1(t_cpx.unsqueeze(1)).squeeze(1)
        t_cpx = self.ln1(t_cpx)

        t_cpx = self.fc6(t_cpx)
        t_cpx = self.avgpool2(t_cpx.unsqueeze(1)).squeeze(1)
        t_cpx = self.ln2(t_cpx)

        t_cpx = self.fc7(t_cpx)                 # [N, pred_len]

        # ---------------- combine complex streams ----------------
        complex_cat = torch.cat((s_complex, t_cpx), dim=1)  # [N, pred_len*2]
        complex_cat = self.dropout(complex_cat)
        complex_proj = self.fc8(complex_cat)               # [N, pred_len]

        # ---------------- adaptive blending with baseline ----------------
        gate_in = torch.cat((complex_proj, baseline_t), dim=1)  # [N, pred_len*2]
        gate = self.sigmoid(self.gate_fc(gate_in))              # [N, pred_len]
        blended = gate * complex_proj + (1.0 - gate) * baseline_t

        # add scaled baseline contributions from c and r
        alpha_c = torch.sigmoid(self.alpha_c).unsqueeze(0)  # [1, pred_len]
        alpha_r = torch.sigmoid(self.alpha_r).unsqueeze(0)
        out = blended + alpha_c * baseline_c + alpha_r * baseline_r

        # reshape back to [B, pred_len, C]
        out = out.reshape(B, C, self.pred_len).permute(0,2,1)
        return out
# ...existing code...