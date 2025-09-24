import math
import torch
import torch.nn.functional as F
from torch import nn

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.seq_len = seq_len

        # Non-linear Stream / patching params
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        # feature dim per patch (kept as original design)
        self.dim = patch_len * patch_len

        # compute expected number of patches (kept fixed per init seq_len)
        self.patch_num = (seq_len - patch_len) // stride + 1
        # if user requested end-padding, keep patch_num constant but pad to match it at runtime
        if padding_patch == 'end':
            # no runtime layer object here; we'll pad in forward with replicate mode
            pass

        # Patch Embedding
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.act = nn.SiLU()
        # normalize across feature dim (stable for small batches)
        self.ln_embed = nn.LayerNorm(self.dim)

        # Depthwise + pointwise convs operate on feature dim as channels.
        # We'll use depthwise conv (groups=self.dim) and GroupNorm(1, dim) for stability.
        self.conv1 = nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim)
        self.gn1 = nn.GroupNorm(1, self.dim)
        self.conv2 = nn.Conv1d(self.dim, self.dim, kernel_size=1)
        self.gn2 = nn.GroupNorm(1, self.dim)

        # Residual projection: keep same dim to add easily
        self.fc2 = nn.Linear(self.dim, self.dim)

        # Flatten head (depends on patch_num and dim)
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * self.dim, pred_len * 2)
        self.act2 = nn.SiLU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # Linear Stream: stable MLP with LayerNorm + SiLU
        self.linear_mlp = nn.Sequential(
            nn.Linear(seq_len, max(seq_len, pred_len * 2)),
            nn.SiLU(),
            nn.LayerNorm(max(seq_len, pred_len * 2)),
            nn.Linear(max(seq_len, pred_len * 2), pred_len)
        )

        # Streams Concatenation
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s, t, c=None, r=None):
        # s,t: [Batch, Input, Channel] from caller
        if c is not None:
            s = s + c
        if r is not None:
            t = t + r

        # to [Batch, Channel, Input]
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        # channel-independent processing: merge batch and channel
        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        s = torch.reshape(s, (B * C, I))  # [B*C, Input]
        t = torch.reshape(t, (B * C, I))  # [B*C, Input]

        # --- Non-linear stream: patching ---
        # pad to ensure consistent number of patches = self.patch_num
        if self.padding_patch == 'end':
            expected_total = self.patch_num * self.stride + self.patch_len - self.stride
            pad_needed = max(0, expected_total - I)
            if pad_needed > 0:
                s = F.pad(s, (0, pad_needed), mode='replicate')

        # unfold into patches: [B*C, patch_num_runtime, patch_len]
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride).contiguous()
        # If runtime patch count differs from init, raise clear error (keeps shapes predictable)
        if s.shape[1] != self.patch_num:
            raise RuntimeError(f"runtime patch_num {s.shape[1]} != init patch_num {self.patch_num}")

        # Patch embedding (applies on last dim)
        s = self.fc1(s)               # [B*C, patch_num, dim]
        s = self.act(s)
        s = self.ln_embed(s)

        # keep residual (same shape)
        res = s

        # prepare for convs: Conv1d expects [N, C, L] where C = feature dim
        s = s.permute(0, 2, 1).contiguous()  # [B*C, dim, patch_num]

        # depthwise conv + GN + act
        s = self.conv1(s)
        s = self.gn1(s)
        s = self.act(s)

        # pointwise conv + GN + act
        s = self.conv2(s)
        s = self.gn2(s)
        s = self.act(s)

        # back to [B*C, patch_num, dim]
        s = s.permute(0, 2, 1).contiguous()

        # residual projection and add
        res = self.fc2(res)  # [B*C, patch_num, dim]
        s = s + res

        # Flatten head -> predict partial output
        s = self.flatten1(s)  # [B*C, patch_num * dim]
        s = self.fc3(s)
        s = self.act2(s)
        s = self.fc4(s)        # [B*C, pred_len]

        # --- Linear stream (reworked MLP) ---
        t = self.linear_mlp(t)  # [B*C, pred_len]

        # --- Combine streams ---
        x = torch.cat((s, t), dim=1)  # [B*C, pred_len*2]
        x = self.fc8(x)               # [B*C, pred_len]

        # restore [Batch, Channel, Output] then permute to [Batch, Output, Channel]
        x = torch.reshape(x, (B, C, self.pred_len))
        x = x.permute(0, 2, 1)

        return x