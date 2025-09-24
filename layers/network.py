# ...existing code...
import math
import torch
import torch.nn.functional as F
from torch import nn

class ComponentFusion(nn.Module):
    """
    Fuse 4 components (seasonal, trend, cyclic, irregular) into one prediction vector.
    Options:
     - branch-per-component: each branch has a small MLP from seq_len -> pred_len
     - gated fusion: compute softmax weights per-sample to combine branches
     - cross-attention: allow branches to attend to each other (lightweight MHA)
    Inputs to forward are flattened per-channel sequences: [N, seq_len] for each component.
    Returns fused output [N, pred_len].
    """
    def __init__(self, seq_len, pred_len, hidden=None, dropout=0.1,
                 use_gated=True, use_cross_attn=True, nhead=4):
        super(ComponentFusion, self).__init__()
        hidden = hidden or max(seq_len, pred_len * 2)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden = hidden
        self.use_gated = use_gated
        self.use_cross_attn = use_cross_attn

        # per-branch encoder -> embedding of sequence (hidden dim)
        self.embed_s = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.embed_t = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.embed_c = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.embed_r = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # optional cross-attention (multi-head attention over 4 tokens)
        if use_cross_attn:
            # MultiheadAttention with batch_first=True where available
            self.cross_attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=nhead, batch_first=True)
            # small feedforward after attention
            self.attn_ff = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        # per-branch output heads mapping hidden -> pred_len
        self.out_s = nn.Linear(hidden, pred_len)
        self.out_t = nn.Linear(hidden, pred_len)
        self.out_c = nn.Linear(hidden, pred_len)
        self.out_r = nn.Linear(hidden, pred_len)

        # gating network: maps concatenated branch embeddings or outputs -> 4 logits
        if use_gated:
            self.gate_net = nn.Sequential(
                nn.Linear(4 * hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 4)
            )

    def forward(self, s, t, c, r):
        # inputs: s,t,c,r: [N, seq_len]
        es = self.embed_s(s)  # [N, hidden]
        et = self.embed_t(t)
        ec = self.embed_c(c)
        er = self.embed_r(r)

        # stack tokens [N, 4, hidden]
        tokens = torch.stack([es, et, ec, er], dim=1)

        if self.use_cross_attn:
            # self-attend across the 4 tokens
            attn_out, _ = self.cross_attn(tokens, tokens, tokens)  # [N,4,hidden]
            attn_out = self.attn_ff(attn_out)
            # split back
            es_att, et_att, ec_att, er_att = attn_out[:,0], attn_out[:,1], attn_out[:,2], attn_out[:,3]
        else:
            es_att, et_att, ec_att, er_att = es, et, ec, er

        # branch outputs [N, pred_len]
        out_s = self.out_s(es_att)
        out_t = self.out_t(et_att)
        out_c = self.out_c(ec_att)
        out_r = self.out_r(er_att)

        # gating: compute sample-wise softmax weights
        if self.use_gated:
            # gate input: concatenated attended embeddings
            gate_in = torch.cat([es_att, et_att, ec_att, er_att], dim=1)  # [N, 4*hidden]
            logits = self.gate_net(gate_in)  # [N,4]
            gates = F.softmax(logits, dim=1)  # [N,4]
            fused = (gates[:, 0:1] * out_s
                   + gates[:, 1:2] * out_t
                   + gates[:, 2:3] * out_c
                   + gates[:, 3:4] * out_r)
        else:
            # simple average
            fused = (out_s + out_t + out_c + out_r) * 0.25

        return fused  # [N, pred_len]


class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, dropout=0.1,
                 fusion_hidden=None, fusion_dropout=0.1, fusion_use_gated=True, fusion_use_attn=True):
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

        # --- Fusion for linear-like stream (component-aware) ---
        self.component_fusion = ComponentFusion(
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            hidden=fusion_hidden,
            dropout=fusion_dropout,
            use_gated=fusion_use_gated,
            use_cross_attn=fusion_use_attn
        )
        # fallback simple linear stream (kept for compatibility)
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
                try:
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                except Exception:
                    pass
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, s, t, c=None, r=None):
        # inputs: s,t: [B, Input, Channel]
        # when MA type is 'learn', model receives (seasonal, trend, cyclic, irregular)
        # c and r may be None for simpler MA types

        if c is not None:
            s = s + c  # keep backwards compatible behavior if caller expects this
        if r is not None:
            t = t + r

        # to [B, C, I]
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        # merge batch and channel for channel-independent processing
        B, C, I = s.shape
        s_flat = s.reshape(B * C, I)  # [N, I]  <-- keep copy for fusion branches
        t_flat = t.reshape(B * C, I)  # [N, I]

        # --- Non-linear stream: patching ---
        s_nl = s_flat  # use copy, will patch below

        if self.padding_patch == 'end':
            expected_total = self.patch_num * self.stride + self.patch_len - self.stride
            pad_needed = max(0, expected_total - I)
            if pad_needed > 0:
                s_nl = F.pad(s_nl, (0, pad_needed), mode='replicate')

        s_patches = s_nl.unfold(dimension=-1, size=self.patch_len, step=self.stride).contiguous()
        if s_patches.shape[1] != self.patch_num:
            raise RuntimeError(f"runtime patch_num {s_patches.shape[1]} != init patch_num {self.patch_num}")

        # patch embedding (applies to last dim)
        s_emb = self.patch_fc(s_patches)             # [N, patch_num, dim]
        s_emb = self.act(s_emb)
        s_emb = self.ln_patch(s_emb)
        s_emb = self.drop(s_emb)

        res = s_emb.clone()

        # prepare for conv: [N, dim, patch_num]
        s_conv = s_emb.permute(0, 2, 1).contiguous()
        s_conv = self.dw_conv(s_conv)
        s_conv = self.gn_dw(s_conv)
        s_conv = self.act(s_conv)

        s_conv = self.pw_conv(s_conv)
        s_conv = self.gn_pw(s_conv)
        s_conv = self.act(s_conv)

        # back to [N, patch_num, dim]
        s_conv = s_conv.permute(0, 2, 1).contiguous()
        s_conv = self.post_ln(s_conv)

        # residual projection and scaled add
        res = self.res_proj(res)
        s_conv = s_conv + self.res_scale * res

        # flatten head -> partial prediction from non-linear stream
        s_head = self.flatten(s_conv)
        s_head = self.head_fc1(s_head)
        s_head = self.head_act(s_head)
        s_head = self.head_fc2(s_head)  # [N, pred_len]

        # --- Component-aware fusion (if cyclic & irregular available) ---
        if c is not None and r is not None:
            # c,r inputs were added to s/t earlier for backwards compatibility.
            # Original components should be provided by caller so we reconstruct flattened versions:
            # We expect incoming s,t,c,r were the per-component series before permute/reshape.
            # At entry s,t are seasonal & trend (possibly modified above); but original c,r tensors
            # must be passed separately if accurate component fusion is desired.
            # Here assume caller passed raw seasonal, trend, cyclic, irregular into this forward call.
            # We rebuild flattened component sequences for fusion from pre-permute variables:
            # NOTE: In standard usage decomp returns seasonal, trend, cyclic, irregular and Model.forward
            # calls net(seasonal, trend, cyclic, irregular) so the variables are already correct.
            comp_s = s_flat
            comp_t = t_flat
            # For cyclic and irregular we need to reconstruct flattened tensors from c/r after permutation
            c_flat = c.permute(0, 2, 1).reshape(B * C, I)
            r_flat = r.permute(0, 2, 1).reshape(B * C, I)
            fused = self.component_fusion(comp_s, comp_t, c_flat, r_flat)  # [N, pred_len]
            t_out = fused
        else:
            # fallback to stable linear MLP on trend
            t_norm = self.t_ln(t_flat)
            t_out = self.linear_mlp(t_norm) + self.linear_skip(t_flat)  # [N, pred_len]

        # --- Combine non-linear and linear/fused streams ---
        x = torch.cat((s_head, t_out), dim=1)  # [N, pred_len*2]
        x = self.combine(x)                    # [N, pred_len]

        # restore shapes [B, C, pred_len] -> [B, pred_len, C]
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)

        return x
# ...existing code...