
import torch
from torch import nn

class Network(nn.Module):
    def __init__(self,
                 seq_len,
                 pred_len,
                 patch_len,
                 stride,
                 padding_patch,
                 merge_sc='add',
                 merge_tr='add',
                 nl_dropout=0.1,
                 linear_dropout=0.1):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.merge_sc = merge_sc  # 'add' or 'concat'
        self.merge_tr = merge_tr  # 'add' or 'concat'

        # Non-linear Stream (patch-based)
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        # compute expected number of patches (will match forward padding behavior)
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':
            self.patch_num += 1

        # Patch Embedding (per-patch MLP)
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.SiLU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)
        self.nl_dropout = nn.Dropout(nl_dropout)

        # Depthwise conv across patch elements
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               kernel_size=patch_len, stride=patch_len, groups=self.patch_num)
        self.gelu2 = nn.SiLU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # Residual stream (project back)
        self.fc2 = nn.Linear(self.dim, patch_len)

        # Pointwise conv across patches
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, kernel_size=1)
        self.gelu3 = nn.SiLU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # Flatten head -> produce pred_len
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.SiLU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # Linear Stream (MLP residual blocks)
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.gelu_l1 = nn.SiLU()
        self.ln_l1 = nn.LayerNorm(pred_len * 4)
        self.linear_dropout = nn.Dropout(linear_dropout)

        self.fc6 = nn.Linear(pred_len * 4, pred_len * 2)
        self.gelu_l2 = nn.SiLU()
        self.ln_l2 = nn.LayerNorm(pred_len * 2)

        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        # residual projection from input to pred_len
        self.linear_res_proj = nn.Linear(seq_len, pred_len)

        # Streams concatenation
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

        # Projections for concat-merge options
        if self.merge_sc == 'concat':
            self.fc_merge_sc = nn.Linear(pred_len * 2, pred_len)
        if self.merge_tr == 'concat':
            self.fc_merge_tr = nn.Linear(pred_len * 2, pred_len)

    def _non_linear_branch(self, x):
        # x: [N, Input] where N = B*C
        # optional end padding: repeat last `stride` values
        if self.padding_patch == 'end':
            pad_len = self.stride
            if pad_len > 0:
                pad = x[:, -pad_len:].clone()
                x = torch.cat((x, pad), dim=-1)

        # extract patches: [N, patch_num, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # per-patch embedding
        x = self.fc1(x)                     # [N, patch_num, dim]
        x = self.gelu1(x)
        # BatchNorm1d expects shape (N, C, L) where C == patch_num
        x = self.bn1(x)
        x = self.nl_dropout(x)

        res = x                             # [N, patch_num, dim]

        # conv1 expects (N, C, L) where C=patch_num, L=patch_len (we set groups accordingly)
        # Here fc1 produced L=self.dim; to align with conv1 kernel, we transpose shape so conv sees L=patch_len
        # We first project res back to patch_len for residual addition after conv
        x = self.conv1(x)                   # [N, patch_num, ?] (kernel=patch_len, stride=patch_len)
        x = self.gelu2(x)
        x = self.bn2(x)

        res = self.fc2(res)                 # [N, patch_num, patch_len]
        # Ensure x and res have same last-dim before adding
        # x from conv1 may have shape [N, patch_num, 1] if kernel==patch_len, so expand/resample accordingly
        if x.shape[-1] != res.shape[-1]:
            # if conv produces length-1 output, squeeze that dim to match patch_len via repeat
            if x.shape[-1] == 1:
                x = x.repeat_interleave(self.patch_len, dim=-1)
            else:
                # fallback: interpolate/resample to match
                x = torch.nn.functional.interpolate(x.permute(0,1,2).contiguous(), size=res.shape[-1], mode='linear', align_corners=False).permute(0,1,2)

        x = x + res

        x = self.conv2(x)
        x = self.gelu3(x)
        x = self.bn3(x)

        x = self.flatten1(x)                # [N, patch_num * patch_len]
        x = self.fc3(x)
        x = self.gelu4(x)
        x = self.fc4(x)                     # [N, pred_len]
        return x

    def _linear_branch(self, x):
        # x: [N, Input] where N = B*C
        skip = self.linear_res_proj(x)      # [N, pred_len]

        x = self.fc5(x)
        x = self.gelu_l1(x)
        x = self.ln_l1(x)
        x = self.linear_dropout(x)

        x = self.fc6(x)
        x = self.gelu_l2(x)
        x = self.ln_l2(x)
        x = self.linear_dropout(x)

        x = self.fc7(x)                     # [N, pred_len]
        x = x + skip
        return x

    def forward(self, s, t, c=None, r=None):
        # Inputs shape: [Batch, Input, Channel]
        s = s.permute(0, 2, 1)  # [B, C, I]
        t = t.permute(0, 2, 1)  # [B, C, I]

        B, C, I = s.shape
        s = torch.reshape(s, (B * C, I))    # [B*C, I]
        t = torch.reshape(t, (B * C, I))    # [B*C, I]

        if c is not None:
            c = c.permute(0, 2, 1)
            c = torch.reshape(c, (B * C, I))
        if r is not None:
            r = r.permute(0, 2, 1)
            r = torch.reshape(r, (B * C, I))

        # Non-linear stream
        s_out = self._non_linear_branch(s)
        if c is not None:
            c_out = self._non_linear_branch(c)
            if self.merge_sc == 'add':
                s_out = s_out + c_out
            else:
                s_out = self.fc_merge_sc(torch.cat((s_out, c_out), dim=1))

        # Linear stream
        t_out = self._linear_branch(t)
        if r is not None:
            r_out = self._linear_branch(r)
            if self.merge_tr == 'add':
                t_out = t_out + r_out
            else:
                t_out = self.fc_merge_tr(torch.cat((t_out, r_out), dim=1))

        # Combine streams and final projection
        x = torch.cat((s_out, t_out), dim=1)  # [B*C, pred_len*2]
        x = self.fc8(x)                       # [B*C, pred_len]

        # reshape back to [Batch, Output, Channel]
        x = torch.reshape(x, (B, C, self.pred_len))
        x = x.permute(0, 2, 1)                # [Batch, pred_len, Channel]
        return x
