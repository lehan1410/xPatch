import math
import torch
import torch.nn.functional as F
from torch import nn

"""
Multiscale -> node encoder + DTW-based memory read

- MultiScaleNodeEncoder: build multiscale temporal features (dilated convs) and soft-pool into K nodes
- PatternMemoryDTW: store M prototype node-sequences; compute DTW between query node-sequence and prototypes,
  convert distances -> soft weights -> read memory values -> fuse + head -> pred_len
- Network: original patch-based non-linear stream + trend -> multiscale nodes -> DTW-memory read -> combine
"""

class MultiScaleNodeEncoder(nn.Module):
    def __init__(self, seq_len, node_dim=64, num_nodes=8, scales=(64, 16, 4), kernel_size=3):
        """
        scales: list of receptive-field-like sizes (larger -> long scale)
        Produces node embeddings per sample: [N, K_total, node_dim]
        """
        super(MultiScaleNodeEncoder, self).__init__()
        self.seq_len = seq_len
        self.node_dim = node_dim
        self.num_nodes = num_nodes
        self.scales = scales
        self.kernel_size = kernel_size

        self.scale_convs = nn.ModuleList()
        for s in scales:
            # build small dilated stack to increase receptive field ~ s
            layers = []
            rf = 1
            dilation = 1
            in_ch = 1
            # ensure subsequent convs accept node_dim channels
            while rf < s:
                layers.append(nn.Conv1d(in_ch, node_dim, kernel_size=kernel_size, padding=dilation, dilation=dilation))
                layers.append(nn.GELU())
                # normalize across feature channels (Conv1d output shape [N, node_dim, L])
                layers.append(nn.BatchNorm1d(node_dim))
                rf += (kernel_size - 1) * dilation
                dilation *= 2
                in_ch = node_dim
            self.scale_convs.append(nn.Sequential(*layers))

        # pooling score network per scale -> produce num_nodes per scale
        self.pool_scores = nn.ModuleList([nn.Linear(node_dim, num_nodes) for _ in scales])


def dtw_distance_vectors(query_nodes, proto_nodes):
    """
    Compute DTW distance between query node-sequence and proto node-sequence.
    query_nodes: [N, Kq, D]
    proto_nodes: [Kp, D] or [N, Kp, D] (if batch-specific prototypes)
    Returns distances [N]
    Uses classic DP; Kq and Kp expected small (~4-32).
    """
    if proto_nodes.dim() == 2:
        # expand to batch
        proto = proto_nodes.unsqueeze(0)  # [1, Kp, D]
    else:
        proto = proto_nodes  # [N, Kp, D]
    N, Kq, D = query_nodes.shape
    Kp = proto.shape[1]

    # compute cost matrix: [N, Kq, Kp]
    q = query_nodes.unsqueeze(2)  # [N, Kq,1,D]
    p = proto.unsqueeze(1)        # [N,1,Kp,D] or [1,1,Kp,D]
    cost = torch.norm(q - p, dim=-1)  # [N, Kq, Kp]

    # DP matrix init
    inf = 1e9
    device = cost.device
    dp = torch.full((N, Kq + 1, Kp + 1), inf, device=device, dtype=cost.dtype)
    dp[:, 0, 0] = 0.0
    # fill
    for i in range(1, Kq + 1):
        for j in range(1, Kp + 1):
            c = cost[:, i - 1, j - 1]
            prev = torch.min(torch.stack([dp[:, i - 1, j], dp[:, i, j - 1], dp[:, i - 1, j - 1]], dim=-1), dim=-1)[0]
            dp[:, i, j] = c + prev
    dist = dp[:, Kq, Kp]  # [N]
    return dist


class PatternMemoryDTW(nn.Module):
    def __init__(self, num_memory=64, node_dim=64, pred_len=96, prototype_k=16, temp=1.0):
        super(PatternMemoryDTW, self).__init__()
        self.num_memory = num_memory
        self.node_dim = node_dim
        self.pred_len = pred_len
        self.prototype_k = prototype_k
        self.temp = temp

        # prototypes: sequences of prototype nodes: [M, K, D]
        self.prototypes = nn.Parameter(torch.randn(num_memory, prototype_k, node_dim) * 0.1)
        # associated memory read vectors (values) mapped to node_dim then to pred_len
        self.memory_values = nn.Parameter(torch.randn(num_memory, node_dim) * 0.1)

        # small projector from fused read -> final pred
        self.read_proj = nn.Sequential(
            nn.Linear(node_dim * 2, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, pred_len)
        )

    def forward(self, nodes):
        """
        nodes: [N, Kq, D] (query node-sequence)
        returns [N, pred_len]
        """
        N, Kq, D = nodes.shape
        M = self.num_memory
        # if prototype length differs from Kq, we can either truncate/pad or compute DTW across different lengths
        # use dtw_distance_vectors which supports different K
        # compute distances [N, M]
        dists = []
        for m in range(M):
            proto = self.prototypes[m]  # [Kp, D]
            dist_m = dtw_distance_vectors(nodes, proto)  # [N]
            dists.append(dist_m)
        dists = torch.stack(dists, dim=1)  # [N, M]

        # similarity weights
        sims = -dists / max(1e-6, self.temp)
        weights = F.softmax(sims, dim=1)  # [N, M]

        # read memory values -> [N, D]
        mem_vals = self.memory_values.unsqueeze(0).expand(N, -1, -1)  # [N, M, D]
        read = torch.einsum('nm,nmd->nd', weights, mem_vals)  # [N, D]

        # fuse read with simple summary of nodes (mean)
        summary = nodes.mean(dim=1)  # [N, D]
        fused = torch.cat([summary, read], dim=1)  # [N, 2D]
        out = self.read_proj(fused)  # [N, pred_len]
        return out, dists  # return distances for diagnostics optionally


class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 node_dim=64, nodes_per_scale=4, scales=(64,16,4),
                 mem_size=64, mem_prototype_k=8):
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
        # use LayerNorm on feature dim (after embedding)
        self.ln1 = nn.LayerNorm(self.dim)

        # CNN Depthwise (operate on feature dim after permute in forward)
        self.conv_dw = nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim)
        self.gelu2 = nn.GELU()
        self.ln2 = nn.LayerNorm(self.dim)

        # Residual Stream
        # self.fc2 = nn.Linear(self.dim, patch_len)
        self.res_proj = nn.Linear(self.dim, self.dim)

        # CNN Pointwise
        self.conv_pw = nn.Conv1d(self.dim, self.dim, kernel_size=1)
        self.gelu3 = nn.GELU()
        self.ln3 = nn.LayerNorm(self.dim)

        # Flatten Head
        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * self.dim, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        # Linear Stream (simple MLP fallback kept)
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln_fc = nn.LayerNorm(pred_len * 2)
        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2b = nn.LayerNorm(pred_len // 2)
        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        # multiscale node encoder for trend (and optionally other comps)
        self.node_encoder = MultiScaleNodeEncoder(seq_len=seq_len,
                                                  node_dim=node_dim,
                                                  num_nodes=nodes_per_scale,
                                                  scales=scales)

        # DTW-based pattern memory
        total_nodes = nodes_per_scale * len(scales)
        self.pattern_mem = PatternMemoryDTW(num_memory=mem_size,
                                            node_dim=node_dim,
                                            pred_len=pred_len,
                                            prototype_k=mem_prototype_k,
                                            temp=1.0)

        # combine streams
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s, t, c=None, r=None):
        # s,t: [Batch, Input, Channel]
        if c is not None:
            s = s + c
        if r is not None:
            t = t + r

        s = s.permute(0,2,1) # to [Batch, Channel, Input]
        t = t.permute(0,2,1) # to [Batch, Channel, Input]

        B = s.shape[0]; C = s.shape[1]; I = s.shape[2]
        s_flat = torch.reshape(s, (B*C, I)) # [N, L]
        t_flat = torch.reshape(t, (B*C, I)) # [N, L]

        # Non-linear Stream (patching)
        if self.padding_patch == 'end':
            s_flat = self.padding_patch_layer(s_flat)
        s_p = s_flat.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [N, patch_num, patch_len]

        s_e = self.fc1(s_p)
        s_e = self.gelu1(s_e)
        s_e = self.ln1(s_e)

        res = s_e
        # convs operate on feature dim: permute to [N, dim, patch_num]
        s_conv = s_e.permute(0,2,1).contiguous()
        s_conv = self.conv_dw(s_conv)
        s_conv = self.gelu2(s_conv)
        s_conv = self.conv_pw(s_conv)
        s_conv = self.gelu3(s_conv)
        s_conv = s_conv.permute(0,2,1).contiguous()
        s_conv = self.ln2(s_conv)

        # res = self.fc2(res)
        res = self.res_proj(res)
        s_conv = s_conv + res

        s_head = self.flatten1(s_conv)
        s_head = self.fc3(s_head)
        s_head = self.gelu4(s_head)
        s_head = self.fc4(s_head)  # [N, pred_len]

        # Node encoding from trend (multiscale)
        nodes = self.node_encoder(t_flat)  # [N, K_total, node_dim]

        # Optionally apply node self-attention to refine nodes (small MHA)
        # Flatten nodes to shape for MHA: treat nodes as sequence
        nodes_attn = nodes  # no-op here; user can add MHA externally

        # DTW memory read
        t_out, dists = self.pattern_mem(nodes_attn)  # [N, pred_len]

        # Streams concat
        x = torch.cat((s_head, t_out), dim=1)
        x = self.fc8(x)

        x = torch.reshape(x, (B, C, self.pred_len)) # [B, C, pred_len]
        x = x.permute(0,2,1) # to [B, pred_len, C]
        return x