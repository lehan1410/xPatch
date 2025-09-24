import math
import torch
import torch.nn.functional as F
from torch import nn

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, dropout=0.1):
        super(Network, self).__init__()

        # params
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        # feature dim per patch (kept quadratic as original design)
        self.dim = patch_len * patch_len

        # computed patch count (kept fixed from init seq_len)
        self.patch_num = (seq_len - patch_len) // stride + 1

        # --- Patch embedding (non-linear stream) ---
        self.patch_fc = nn.Linear(patch_len, self.dim)
        self.act = nn.SiLU()
        self.ln_patch = nn.LayerNorm(self.dim)
        self.drop = nn.Dropout(dropout)

        # Depthwise conv on feature dim (groups=self.dim) + pointwise conv
        # Conv1d expects [N, C, L] where C = feature dim
        self.dw_conv = nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim)
        self.gn_dw = nn.GroupNorm(1, self.dim)
        self.pw_conv = nn.Conv1d(self.dim, self.dim, kernel_size=1)
        self.gn_pw = nn.GroupNorm(1, self.dim)

        # small post conv LayerNorm (applied on last dim)
        self.post_ln = nn.LayerNorm(self.dim)

        # residual projection (same shape so easy add) + small learnable scale
        self.res_proj = nn.Linear(self.dim, self.dim)
        self.res_scale = nn.Parameter(torch.tensor(0.5))

        # flatten head to predict pred_len
        self.flatten = nn.Flatten(start_dim=-2)
        self.head_fc1 = nn.Linear(self.patch_num * self.dim, pred_len * 2)
        self.head_act = nn.SiLU()
        self.head_fc2 = nn.Linear(pred_len * 2, pred_len)

        # --- Linear stream (stable MLP) ---
        hidden = max(self.seq_len, pred_len * 2)
        self.t_ln = nn.LayerNorm(self.seq_len)
        self.linear_mlp = nn.Sequential(
            nn.Linear(self.seq_len, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, pred_len)
        )
        self.linear_skip = nn.Linear(self.seq_len, pred_len)

        # combine streams
        self.combine = nn.Linear(pred_len * 2, pred_len)

        # init
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, s, t, c=None, r=None):
        # inputs: s,t: [B, Input, Channel]
        if c is not None:
            s = s + c
        if r is not None:
            t = t + r

        # to [B, C, I]
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        # merge batch and channel for channel-independent processing
        B, C, I = s.shape
        s = s.reshape(B * C, I)  # [N, I]
        t = t.reshape(B * C, I)  # [N, I]

        # --- Non-linear stream: patching ---
        if self.padding_patch == 'end':
            expected_total = self.patch_num * self.stride + self.patch_len - self.stride
            pad_needed = max(0, expected_total - I)
            if pad_needed > 0:
                s = F.pad(s, (0, pad_needed), mode='replicate')

        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride).contiguous()
        if s.shape[1] != self.patch_num:
            raise RuntimeError(f"runtime patch_num {s.shape[1]} != init patch_num {self.patch_num}")

        # patch embedding (applies to last dim)
        s = self.patch_fc(s)             # [N, patch_num, dim]
        s = self.act(s)
        s = self.ln_patch(s)
        s = self.drop(s)

        res = s.clone()

        # prepare for conv: [N, dim, patch_num]
        s = s.permute(0, 2, 1).contiguous()
        s = self.dw_conv(s)
        s = self.gn_dw(s)
        s = self.act(s)

        s = self.pw_conv(s)
        s = self.gn_pw(s)
        s = self.act(s)

        # back to [N, patch_num, dim]
        s = s.permute(0, 2, 1).contiguous()
        s = self.post_ln(s)

        # residual projection and scaled add
        res = self.res_proj(res)
        s = s + self.res_scale * res

        # flatten head -> partial prediction
        s = self.flatten(s)
        s = self.head_fc1(s)
        s = self.head_act(s)
        s = self.head_fc2(s)  # [N, pred_len]

        # --- Linear stream ---
        t_norm = self.t_ln(t)
        t_out = self.linear_mlp(t_norm) + self.linear_skip(t)  # [N, pred_len]

        # --- Combine ---
        x = torch.cat((s, t_out), dim=1)  # [N, pred_len*2]
        x = self.combine(x)               # [N, pred_len]

        # restore shapes [B, C, pred_len] -> [B, pred_len, C]
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)

        return x