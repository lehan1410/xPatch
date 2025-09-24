import math
import torch
import torch.nn.functional as F
from torch import nn

class HypergraphMemory(nn.Module):
    """
    Build hypernodes by soft-pooling over time, run node self-attention,
    then query a dynamic memory (DMN style) and produce a prediction vector.
    Input: t_flat [N, seq_len] (and optional other comps concatenated)
    Output: [N, pred_len]
    """
    def __init__(self, seq_len, pred_len, num_nodes=8, node_dim=128, memory_size=64, memory_dim=128, nhead=4, dropout=0.1):
        super(HypergraphMemory, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.memory_size = memory_size
        self.memory_dim = memory_dim

        # local temporal encoder -> produce per-time features
        self.time_enc = nn.Sequential(
            nn.Conv1d(1, node_dim, kernel_size=3, padding=1),
            nn.GELU(),
            # LayerNorm over channel dimension; using LayerNorm with shape tuple for conv output
            nn.LayerNorm([node_dim, seq_len])
        )

        # compute pooling scores for nodes from per-time features
        self.pool_score = nn.Linear(node_dim, num_nodes)

        # node self-attention (cross-node interactions)
        self.node_attn = nn.MultiheadAttention(embed_dim=node_dim, num_heads=nhead, batch_first=True)
        self.node_ff = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Dynamic Memory (learnable keys)
        self.memory = nn.Parameter(torch.randn(memory_size, memory_dim) * 0.1)
        # project node-level summary to memory query dim
        self.q_proj = nn.Linear(node_dim, memory_dim)
        self.read_ff = nn.Sequential(
            nn.Linear(memory_dim + node_dim, node_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # final head from fused representation -> pred_len
        self.head = nn.Sequential(
            nn.Linear(node_dim, max(node_dim // 2, 8)),
            nn.GELU(),
            nn.Linear(max(node_dim // 2, 8), pred_len)
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t_flat):
        # t_flat: [N, seq_len]
        N, L = t_flat.shape
        x = t_flat.unsqueeze(1)  # [N,1,L]
        # time features: [N, node_dim, L]
        tf = self.time_enc(x)
        # transpose to [N, L, node_dim]
        tf_t = tf.permute(0, 2, 1).contiguous()

        # pooling scores -> [N, L, num_nodes]
        scores = self.pool_score(tf_t)  # per-time scores for each node
        scores = F.softmax(scores, dim=1)  # soft pool over time (sum to 1 over L)
        # produce nodes by weighted sum over time: [N, num_nodes, node_dim]
        nodes = torch.einsum('nlk,nld->nkd', scores, tf_t)

        # node self-attention
        attn_out, _ = self.node_attn(nodes, nodes, nodes)  # [N, num_nodes, node_dim]
        nodes = nodes + attn_out
        nodes = nodes + self.node_ff(nodes)

        # produce graph summary by mean pooling nodes -> [N, node_dim]
        graph_summary = nodes.mean(dim=1)

        # query memory
        q = self.q_proj(graph_summary)  # [N, memory_dim]
        # similarity and read
        mem = self.memory.unsqueeze(0).expand(N, -1, -1)  # [N, memory_size, memory_dim]
        sims = torch.einsum('nd,nmd->nm', q, mem)
        weights = F.softmax(sims / math.sqrt(max(1, self.memory_dim)), dim=1)  # [N, memory_size]
        read = torch.einsum('nm,nmd->nd', weights, mem)  # [N, memory_dim]

        # fuse read with graph summary -> node_dim
        if read.shape[1] != graph_summary.shape[1]:
            proj = nn.Linear(read.shape[1], graph_summary.shape[1]).to(read.device)
            read = proj(read)
        fused = self.read_ff(torch.cat([graph_summary, read], dim=1))

        # predict
        out = self.head(fused)  # [N, pred_len]
        return out


class DMNAdapter(nn.Module):
    """
    Adapter that builds hypernodes by soft pooling (like HypergraphMemory),
    then calls an external DMN module and reduces outputs to pred_len.
    dmn_module.forward(nodes, [time_node_embs, global_emb]) is expected.
    """
    def __init__(self, seq_len, pred_len, num_nodes, node_dim, dmn_module=None):
        super(DMNAdapter, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.dmn = dmn_module

        self.time_enc = nn.Sequential(
            nn.Conv1d(1, node_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.LayerNorm([node_dim, seq_len])
        )
        self.pool_score = nn.Linear(node_dim, num_nodes)
        # learnable global node embedding used as second argument to DMN
        self.global_node_emb = nn.Parameter(torch.randn(num_nodes, node_dim) * 0.05)
        # readout if DMN returns node-wise features
        self.readout = nn.Linear(node_dim, pred_len)

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

    def forward(self, t_flat):
        # t_flat: [N, L]
        N, L = t_flat.shape
        x = t_flat.unsqueeze(1)                       # [N,1,L]
        tf = self.time_enc(x)                         # [N, node_dim, L]
        tf_t = tf.permute(0, 2, 1).contiguous()       # [N, L, node_dim]
        scores = self.pool_score(tf_t)                # [N, L, num_nodes]
        scores = F.softmax(scores, dim=1)
        nodes = torch.einsum('nlk,nld->nkd', scores, tf_t)  # [N, num_nodes, node_dim]

        if self.dmn is None:
            # fallback: average nodes and map
            pooled = nodes.mean(dim=1)               # [N, node_dim]
            return self.readout(pooled)

        # DMN expected inputs: nodes [N,num_nodes,node_dim], node_embeddings list
        dmn_out = self.dmn(nodes, [nodes, self.global_node_emb])  # adapt signature to user's DMN
        # dmn_out may be [N, num_nodes, D] or [N, pred_len]
        if dmn_out.dim() == 3:
            pooled = dmn_out.mean(dim=1)
        else:
            pooled = dmn_out
        # ensure final shape
        if pooled.shape[1] != self.pred_len:
            out = self.readout(pooled)
        else:
            out = pooled
        return out


class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 dropout=0.1, hg_nodes=8, hg_node_dim=128, mem_size=64, mem_dim=128,
                 use_dmn=False, dmn_module=None):
        super(Network, self).__init__()

        # params
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        # feature dim per patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1

        # Non-linear stream (patch based)
        self.fc1 = nn.Linear(patch_len, self.dim)
        self.act = nn.GELU()
        self.ln1 = nn.LayerNorm(self.dim)
        self.dw_conv = nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim)
        self.pw_conv = nn.Conv1d(self.dim, self.dim, kernel_size=1)
        self.post_ln = nn.LayerNorm(self.dim)
        self.res_proj = nn.Linear(self.dim, self.dim)
        self.res_scale = nn.Parameter(torch.tensor(0.5))

        self.flatten = nn.Flatten(start_dim=-2)
        self.head_fc1 = nn.Linear(self.patch_num * self.dim, pred_len * 2)
        self.head_fc2 = nn.Linear(pred_len * 2, pred_len)

        # Pattern memory / Hypergraph memory for linear-like stream
        if use_dmn and (dmn_module is not None):
            self.hgmem = DMNAdapter(seq_len=seq_len, pred_len=pred_len,
                                    num_nodes=hg_nodes, node_dim=hg_node_dim,
                                    dmn_module=dmn_module)
        else:
            self.hgmem = HypergraphMemory(seq_len=seq_len, pred_len=pred_len,
                                          num_nodes=hg_nodes, node_dim=hg_node_dim,
                                          memory_size=mem_size, memory_dim=mem_dim)

        # fallback simple linear MLP (kept for compatibility)
        hidden = max(seq_len, pred_len * 2)
        self.t_ln = nn.LayerNorm(seq_len)
        self.linear_mlp = nn.Sequential(
            nn.Linear(seq_len, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, pred_len)
        )
        self.linear_skip = nn.Linear(seq_len, pred_len)

        # combine streams
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
        # inputs: s,t: [B, Input, Channel]
        if c is not None:
            s = s + c
        if r is not None:
            t = t + r

        # to [B, C, I]
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        B, C, I = s.shape
        s_flat = s.reshape(B * C, I)
        t_flat = t.reshape(B * C, I)

        # --- Non-linear stream ---
        if self.padding_patch == 'end':
            expected_total = self.patch_num * self.stride + self.patch_len - self.stride
            pad_needed = max(0, expected_total - I)
            if pad_needed > 0:
                s_flat = F.pad(s_flat, (0, pad_needed), mode='replicate')

        s_patches = s_flat.unfold(dimension=-1, size=self.patch_len, step=self.stride).contiguous()
        if s_patches.shape[1] != self.patch_num:
            raise RuntimeError(f"runtime patch_num {s_patches.shape[1]} != init patch_num {self.patch_num}")

        s_emb = self.fc1(s_patches)
        s_emb = self.act(s_emb)
        s_emb = self.ln1(s_emb)

        res = s_emb.clone()
        s_conv = s_emb.permute(0, 2, 1).contiguous()
        s_conv = self.dw_conv(s_conv)
        s_conv = self.act(s_conv)
        s_conv = self.pw_conv(s_conv)
        s_conv = self.act(s_conv)
        s_conv = s_conv.permute(0, 2, 1).contiguous()
        s_conv = self.post_ln(s_conv)

        res = self.res_proj(res)
        s_head = s_conv + self.res_scale * res

        s_head = self.flatten(s_head)
        s_head = self.head_fc1(s_head)
        s_head = self.act(s_head)
        s_head = self.head_fc2(s_head)  # [N, pred_len]

        # --- Hypergraph / DMN stream ---
        try:
            t_out = self.hgmem(t_flat)  # [N, pred_len]
        except Exception:
            t_norm = self.t_ln(t_flat)
            t_out = self.linear_mlp(t_norm) + self.linear_skip(t_flat)

        # --- Combine ---
        x = torch.cat((s_head, t_out), dim=1)
        x = self.combine(x)

        # restore shapes
        x = x.reshape(B, C, self.pred_len).permute(0, 2, 1)
        return x