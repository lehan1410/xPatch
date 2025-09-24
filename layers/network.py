import torch
import torch.nn.functional as F
from torch import nn

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, dropout=0.1):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.seq_len = seq_len

        # Non-linear Stream / patching params
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        # feature dim per patch
        self.dim = patch_len * patch_len

        # compute expected number of patches
        self.patch_num = (seq_len - patch_len) // stride + 1

        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.act = nn.SiLU()
        self.ln_embed = nn.LayerNorm(self.dim)
        self.drop = nn.Dropout(dropout)

        # Depthwise (per-feature) conv + pointwise conv
        # operate on feature dim as channels: Conv1d expects [N, C, L]
        self.conv1 = nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim)
        self.conv2 = nn.Conv1d(self.dim, self.dim, kernel_size=1)
        self.post_ln = nn.LayerNorm(self.dim)

        # Residual projection (keep same dim)
        self.fc2 = nn.Linear(self.dim, self.dim)
        self.res_scale = nn.Parameter(torch.tensor(0.5))

        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * self.dim, pred_len * 2)
        self.act2 = nn.SiLU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # Linear Stream: stable MLP with LayerNorm + SiLU + skip
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

        # Streams Concatenation
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

        # init
        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5)) if hasattr(nn.init, 'kaiming_uniform_') else None
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, s, t, c=None, r=None):
        # s,t: [Batch, Input, Channel]
        if c is not None:
            s = s + c
        if r is not None:
            t = t + r

        # to [Batch, Channel, Input]
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        # merge batch and channel
        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        s = torch.reshape(s, (B * C, I))
        t = torch.reshape(t, (B * C, I))

        # --- Non-linear stream: patching ---
        if self.padding_patch == 'end':
            expected_total = self.patch_num * self.stride + self.patch_len - self.stride
            pad_needed = max(0, expected_total - I)
            if pad_needed > 0:
                s = F.pad(s, (0, pad_needed), mode='replicate')

        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride).contiguous()
        if s.shape[1] != self.patch_num:
            raise RuntimeError(f"runtime patch_num {s.shape[1]} != init patch_num {self.patch_num}")

        # Patch embedding
        s = self.fc1(s)               # [B*C, patch_num, dim]
        s = self.act(s)
        s = self.ln_embed(s)
        s = self.drop(s)

        res = s                       # residual

        # conv expects [N, C, L] where C = dim
        s = s.permute(0, 2, 1).contiguous()  # [B*C, dim, patch_num]
        s = self.conv1(s)
        s = self.act(s)
        s = self.conv2(s)
        s = self.act(s)
        s = s.permute(0, 2, 1).contiguous()   # [B*C, patch_num, dim]

        s = self.post_ln(s)

        # residual projection with scale
        res = self.fc2(res)
        s = s + self.res_scale * res

        # Flatten head
        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.act2(s)
        s = self.fc4(s)   # [B*C, pred_len]

        # --- Linear stream ---
        t_norm = self.t_ln(t)
        t_out = self.linear_mlp(t_norm) + self.linear_skip(t)

        # --- Combine ---
        x = torch.cat((s, t_out), dim=1)
        x = self.fc8(x)

        # restore shapes
        x = torch.reshape(x, (B, C, self.pred_len))
        x = x.permute(0, 2, 1)

        return x