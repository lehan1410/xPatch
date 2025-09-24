import torch
from torch import nn

class Network(nn.Module):
    """
    Encoder-Decoder Transformer with sequence-aware memory and autoregressive decoder.
    - Encoder produces a memory sequence from non-linear (patch) and linear (per-step) encoders.
    - Decoder autoregressively generates pred_len steps; when `target` is provided during training,
      teacher forcing is applied (uses ground-truth next-step embeddings).
    Inputs: s,t,(optional)c,(optional)r shapes = [B, seq_len, C]
    Optional: target shape = [B, pred_len, C]
    Output: [B, pred_len, C]
    """
    def __init__(self,
                 seq_len,
                 pred_len,
                 patch_len,
                 stride,
                 padding_patch,
                 d_model=128,
                 nhead=8,
                 num_decoder_layers=3,
                 dim_feedforward=256,
                 dropout=0.1,
                 merge_sc='add',
                 merge_tr='add',
                 nl_dropout=0.1,
                 linear_dropout=0.1,
                 seq_module='tcn'):
        super(Network, self).__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # config
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.merge_sc = merge_sc
        self.merge_tr = merge_tr
        self.d_model = d_model
        self.seq_module = seq_module

        # patching params
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':
            self.patch_num += 1
        self.dim_patch = patch_len * patch_len

        # optional sequence preprocessor (light TCN)
        if self.seq_module == 'tcn':
            tcn_layers = []
            dilations = [1, 2, 4]
            for d in dilations:
                tcn_layers.append(nn.Conv1d(1, 1, kernel_size=3, padding=d, dilation=d))
                tcn_layers.append(nn.SiLU())
            self.tcn = nn.Sequential(*tcn_layers)
        else:
            self.tcn = None

        # ---- Non-linear (patch) encoder producing sequence of patch embeddings ----
        # per-patch embedding (maps patch_len -> dim_patch), then project per-patch to d_model
        self.patch_fc = nn.Linear(self.patch_len, self.dim_patch)
        self.patch_act = nn.SiLU()
        self.patch_dropout = nn.Dropout(nl_dropout)
        # optional conv mixing across patch elements (kept from previous)
        self.patch_conv = nn.Conv1d(self.patch_num, self.patch_num,
                                    kernel_size=self.patch_len, stride=self.patch_len, groups=self.patch_num)
        self.patch_conv_act = nn.SiLU()
        # project each patch embedding to d_model (per-patch)
        self.patch_proj = nn.Linear(self.dim_patch, d_model)

        # ---- Linear encoder (per-timestep projection -> sequence of d_model) ----
        # project scalar/value -> d_model for each time step
        self.lin_time_proj = nn.Linear(1, d_model)
        # small MLP to refine and residual projection
        self.lin_fc1 = nn.Linear(d_model, dim_feedforward)
        self.lin_act1 = nn.SiLU()
        self.lin_ln1 = nn.LayerNorm(dim_feedforward)
        self.lin_dropout = nn.Dropout(linear_dropout)
        self.lin_fc2 = nn.Linear(dim_feedforward, d_model)
        self.lin_res_proj = nn.Linear(1, d_model)  # per-timestep residual proj

        # If merge via concat we need projection back to d_model
        if self.merge_sc == 'concat':
            self.merge_sc_proj = nn.Linear(d_model * 2, d_model)
        if self.merge_tr == 'concat':
            self.merge_tr_proj = nn.Linear(d_model * 2, d_model)

        # ---- Transformer Decoder ----
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model,
                                                   nhead=nhead,
                                                   dim_feedforward=dim_feedforward,
                                                   dropout=dropout,
                                                   activation='gelu')
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # learned start token and learned queries not used (we autoregressively build tgt)
        self.start_token = nn.Parameter(torch.randn(d_model))

        # map scalar prediction -> embedding for feeding as next-step input
        self.output_to_emb = nn.Linear(1, d_model)

        # final head: map d_model -> 1 (scalar per time-step per channel)
        self.head = nn.Linear(d_model, 1)

        # small projection after combining encoders
        self.combine_proj = nn.Linear(d_model, d_model)

    def _seq_preprocess(self, x):
        # x: [N, I]
        if self.tcn is None:
            return x
        x_ch = x.unsqueeze(1)  # [N,1,I]
        res = x_ch
        out = self.tcn(x_ch)
        out = out + res
        return out.squeeze(1)

    def _nonlinear_encode_seq(self, x):
        # x: [N, I] -> produce [N, patch_num, d_model]
        # optional end padding
        if self.padding_patch == 'end':
            pad_len = self.stride
            if pad_len > 0:
                pad = x[:, -pad_len:].clone()
                x = torch.cat((x, pad), dim=-1)
        # unfold -> [N, patch_num, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # per-patch embedding -> [N, patch_num, dim_patch]
        x = self.patch_fc(x)
        x = self.patch_act(x)
        x = self.patch_dropout(x)
        # optional conv mixing (expects input shape [N, C, L], we transpose)
        # we keep conv but it's optional; guard by shapes
        try:
            x_conv = self.patch_conv(x)  # may produce different last-dim; ignore content-heavy mixing
            x_conv = self.patch_conv_act(x_conv)
            # if conv changed last dim, map back by interpolation/resizing - but easier: ignore and use x_conv if shape matches
            if x_conv.shape == x.shape:
                x = x_conv
        except Exception:
            # fallback: skip conv if incompatible
            pass
        # project per-patch to d_model -> [N, patch_num, d_model]
        x = self.patch_proj(x)
        return x

    def _linear_encode_seq(self, x):
        # x: [N, I] -> produce [N, I, d_model]
        # project each scalar per time-step
        x_ts = x.unsqueeze(-1)                # [N, I, 1]
        emb = self.lin_time_proj(x_ts)        # [N, I, d_model]
        # refine per-step with small MLP + residual from scalar->d_model
        res = self.lin_res_proj(x_ts)         # [N, I, d_model]
        h = self.lin_fc1(emb)                 # [N, I, dim_feedforward]
        h = self.lin_act1(h)
        h = self.lin_ln1(h)
        h = self.lin_dropout(h)
        h = self.lin_fc2(h)                   # [N, I, d_model]
        out = h + res
        return out

    def forward(self, s, t, c=None, r=None, target=None, teacher_forcing=True):
        # Inputs: s,t shapes [B, seq_len, C]; target optional [B, pred_len, C]
        B, I, C = s.shape
        assert I == self.seq_len, "input seq len mismatch"
        N = B * C

        # reshape per-channel as batch dimension for encoders: [B*C, I]
        s_in = s.reshape(N, I)
        t_in = t.reshape(N, I)
        c_in = c.reshape(N, I) if c is not None else None
        r_in = r.reshape(N, I) if r is not None else None

        # optional seq preprocessing (TCN)
        if self.tcn is not None:
            s_in = self._seq_preprocess(s_in)
            t_in = self._seq_preprocess(t_in)
            if c_in is not None:
                c_in = self._seq_preprocess(c_in)
            if r_in is not None:
                r_in = self._seq_preprocess(r_in)

        # encode sequences
        s_seq = self._nonlinear_encode_seq(s_in)      # [N, patch_num, d_model]
        if c_in is not None:
            c_seq = self._nonlinear_encode_seq(c_in)
            if self.merge_sc == 'add':
                s_seq = s_seq + c_seq
            else:
                # concat along feature dim then project back to d_model (simple approach: average project)
                s_seq = (s_seq + c_seq) * 0.5

        t_seq = self._linear_encode_seq(t_in)         # [N, I, d_model]
        if r_in is not None:
            r_seq = self._linear_encode_seq(r_in)
            if self.merge_tr == 'add':
                t_seq = t_seq + r_seq
            else:
                t_seq = (t_seq + r_seq) * 0.5

        # combine patch sequence and time-step sequence into memory sequence
        # first ensure same feature dim (d_model), then concatenate along sequence dim
        # s_seq: [N, patch_num, d_model], t_seq: [N, I, d_model] -> memory_seq: [N, patch_num+I, d_model]
        memory_seq = torch.cat([s_seq, t_seq], dim=1)   # [N, Lm, d_model]
        # optional combine projection per token
        memory_seq = self.combine_proj(memory_seq)      # [N, Lm, d_model]

        # Transformer expects memory: (S, N, E)
        memory = memory_seq.permute(1, 0, 2).contiguous()  # [Lm, N, d_model]

        # prepare autoregressive decoding
        device = s.device
        # prepare target ground-truth values per channel if provided
        if target is not None:
            # target shape [B, pred_len, C] -> [N, pred_len]
            target_n = target.permute(0, 2, 1).reshape(N, self.pred_len)
        else:
            target_n = None

        # initial tgt sequence: start token
        start = self.start_token.to(device)
        cur_tgt = start.unsqueeze(0).unsqueeze(1).expand(1, N, self.d_model).contiguous()  # [1, N, d_model]

        outputs = []
        for step in range(self.pred_len):
            # feed current tgt sequence into decoder -> (Tcur, N, d_model)
            dec_out = self.transformer_decoder(tgt=cur_tgt, memory=memory)  # [Tcur, N, d_model]
            last = dec_out[-1]  # [N, d_model]
            pred_scalar = self.head(last).squeeze(-1)  # [N]
            outputs.append(pred_scalar)

            # prepare next input embedding
            if self.training and teacher_forcing and (target_n is not None):
                # use ground-truth next value as teacher forcing
                next_val = target_n[:, step].unsqueeze(-1)  # [N,1]
                next_emb = self.output_to_emb(next_val)     # [N, d_model]
            else:
                # use own prediction
                next_val = pred_scalar.unsqueeze(-1)        # [N,1]
                next_emb = self.output_to_emb(next_val)     # [N, d_model]

            # append next_emb as new time-step in tgt sequence
            next_emb = next_emb.unsqueeze(0)  # [1, N, d_model]
            cur_tgt = torch.cat([cur_tgt, next_emb], dim=0)  # [Tcur+1, N, d_model]

        # stack outputs -> [pred_len, N] -> reshape to [B, pred_len, C]
        out_stack = torch.stack(outputs, dim=0)        # [pred_len, N]
        out_stack = out_stack.permute(1, 0)            # [N, pred_len]
        out_stack = out_stack.reshape(B, C, self.pred_len)  # [B, C, pred_len]
        out_stack = out_stack.permute(0, 2, 1)         # [B, pred_len, C]
        return out_stack