import math
import torch
import torch.nn.functional as F
from torch import nn

class TCNEncoder(nn.Module):
    """
    Lightweight TCN encoder for linear/trend stream.
    Input: x [N, L] -> unsqueeze -> [N, 1, L] -> stack of dilated convs -> pooled -> [N, hidden]
    """
    def __init__(self, seq_len, hidden=128, kernel_size=3, num_layers=4, dropout=0.1):
        super(TCNEncoder, self).__init__()
        self.seq_len = seq_len
        self.hidden = hidden
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        in_ch = 1
        dilation = 1
        for i in range(num_layers):
            out_ch = hidden
            pad = (kernel_size - 1) * dilation // 2
            self.convs.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad, dilation=dilation))
            self.convs.append(nn.GroupNorm(1, out_ch))
            self.convs.append(nn.GELU())
            self.convs.append(nn.Dropout(dropout))
            in_ch = out_ch
            dilation *= 2
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                try:
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                except Exception:
                    pass
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [N, L]
        x = x.unsqueeze(1)            # [N,1,L]
        for layer in self.convs:
            x = layer(x)
        x = self.pool(x).squeeze(-1)  # [N, hidden]
        x = self.out_proj(x)          # [N, hidden]
        return x


class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 nl_dim=None, tcn_hidden=None, tcn_layers=4, dropout=0.1):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        # Non-linear stream params
        self.dim = (nl_dim or max(64, patch_len * 2))
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':
            # pad in forward so keep patch_num as init computed
            pass

        # --- Non-linear (patch) stream ---
        # Patch embedding: map patch_len -> dim
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.act = nn.SiLU()
        self.ln_patch = nn.LayerNorm(self.dim)
        # Depthwise conv across patch positions (feature-wise depthwise)
        self.dw_conv = nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim)
        self.gn_dw = nn.GroupNorm(1, self.dim)
        # Pointwise conv to mix features
        self.pw_conv = nn.Conv1d(self.dim, self.dim, kernel_size=1)
        self.gn_pw = nn.GroupNorm(1, self.dim)
        # residual projection to match dims
        self.res_proj = nn.Linear(self.dim, self.dim)
        self.res_scale = nn.Parameter(torch.tensor(0.5))
        # head to predict
        self.flatten = nn.Flatten(start_dim=-2)
        self.head_fc1 = nn.Linear(self.patch_num * self.dim, pred_len * 2)
        self.head_fc2 = nn.Linear(pred_len * 2, pred_len)

        # --- Linear / trend stream (TCN) ---
        tcn_h = tcn_hidden or max(64, pred_len * 2)
        self.tcn = TCNEncoder(seq_len=seq_len, hidden=tcn_h, num_layers=tcn_layers, dropout=dropout)
        # small MLP on top of TCN + skip linear from input
        self.trend_mlp = nn.Sequential(
            nn.Linear(tcn_h, tcn_h // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(tcn_h // 2, pred_len)
        )
        self.trend_skip = nn.Linear(seq_len, pred_len)

        # Combine streams
        self.combine = nn.Linear(pred_len * 2, pred_len)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                try:
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                except Exception:
                    pass
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, s, t, c=None, r=None):
        # s,t: [B, Input, C] per original convention (seasonal, trend)
        if c is not None:
            s = s + c
        if r is not None:
            t = t + r

        # permute to [B, C, L]
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        B, C, L = s.shape
        s_flat = s.reshape(B * C, L)   # [N, L]
        t_flat = t.reshape(B * C, L)   # [N, L]

        # --- Non-linear patch stream ---
        # padding for patches if needed
        if self.padding_patch == 'end':
            expected_total = self.patch_num * self.stride + self.patch_len - self.stride
            pad_needed = max(0, expected_total - L)
            if pad_needed > 0:
                s_flat = F.pad(s_flat, (0, pad_needed), mode='replicate')

        s_patches = s_flat.unfold(dimension=-1, size=self.patch_len, step=self.stride).contiguous()
        if s_patches.shape[1] != self.patch_num:
            # allow dynamic patch_num by recomputing head shapes if necessary
            raise RuntimeError(f"runtime patch_num {s_patches.shape[1]} != init patch_num {self.patch_num}")

        s_emb = self.fc1(s_patches)     # [N, patch_num, dim]
        s_emb = self.act(s_emb)
        s_emb = self.ln_patch(s_emb)

        res = s_emb.clone()             # residual in [N, patch_num, dim]

        # conv expects [N, dim, patch_num]
        s_conv = s_emb.permute(0, 2, 1).contiguous()
        s_conv = self.dw_conv(s_conv)
        s_conv = self.gn_dw(s_conv)
        s_conv = self.act(s_conv)

        s_conv = self.pw_conv(s_conv)
        s_conv = self.gn_pw(s_conv)
        s_conv = self.act(s_conv)

        s_conv = s_conv.permute(0, 2, 1).contiguous()  # [N, patch_num, dim]
        # residual projection and scaled add
        res = self.res_proj(res)
        s_conv = s_conv + self.res_scale * res

        # flatten head -> partial prediction from non-linear stream
        s_head = self.flatten(s_conv)
        s_head = self.head_fc1(s_head)
        s_head = self.act(s_head)
        s_head = self.head_fc2(s_head)  # [N, pred_len]

        # --- Linear / trend stream (TCN + MLP + skip) ---
        t_feat = self.tcn(t_flat)                # [N, tcn_hidden]
        t_out = self.trend_mlp(t_feat) + self.trend_skip(t_flat)  # [N, pred_len]

        # --- Combine streams ---
        x = torch.cat((s_head, t_out), dim=1)    # [N, pred_len*2]
        x = self.combine(x)                      # [N, pred_len]

        # restore shape [B*C, pred_len] -> [B, pred_len, C]
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)
        return x