# ...existing code...
import torch
from torch import nn

class Network(nn.Module):
    """
    Encoder-Decoder architecture:
      - two lightweight encoders (non-linear patch encoder + linear MLP encoder)
      - combine encoder latents per channel into memory
      - Transformer decoder with learned target queries (length = pred_len)
      - final linear head projects decoder outputs -> scalar per channel per time-step
    Inputs expected: s,t,(optional)c,(optional)r shapes = [B, seq_len, C]
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

        # ---- Non-linear (patch) encoder ----
        # per-patch embedding then flatten -> project to d_model
        self.patch_fc = nn.Linear(self.patch_len, self.dim_patch)
        self.patch_act = nn.SiLU()
        # simple convs retained for local mixing
        self.patch_conv = nn.Conv1d(self.patch_num, self.patch_num,
                                    kernel_size=self.patch_len, stride=self.patch_len, groups=self.patch_num)
        self.patch_conv_act = nn.SiLU()
        # projection from flattened patches -> d_model
        self.patch_proj = nn.Linear(self.patch_num * self.patch_len, d_model)
        self.patch_dropout = nn.Dropout(nl_dropout)

        # ---- Linear encoder (MLP residual) ----
        self.lin_fc1 = nn.Linear(seq_len, dim_feedforward)
        self.lin_act1 = nn.SiLU()
        self.lin_ln1 = nn.LayerNorm(dim_feedforward)
        self.lin_dropout = nn.Dropout(linear_dropout)
        self.lin_fc2 = nn.Linear(dim_feedforward, d_model)
        # residual projection from input -> d_model (for stability)
        self.lin_res_proj = nn.Linear(seq_len, d_model)

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

        # learned target queries (pred_len x d_model)
        self.tgt_queries = nn.Parameter(torch.randn(pred_len, d_model))

        # final head: map d_model -> 1 (scalar per time-step per channel)
        self.head = nn.Linear(d_model, 1)

        # final projection if we concat non-linear+linear encodings before forming memory
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

    def _nonlinear_encode(self, x):
        # x: [N, I]
        # optional end padding
        if self.padding_patch == 'end':
            pad_len = self.stride
            if pad_len > 0:
                pad = x[:, -pad_len:].clone()
                x = torch.cat((x, pad), dim=-1)
        # unfold -> [N, patch_num, patch_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # per-patch embedding
        x = self.patch_fc(x)                 # [N, patch_num, dim_patch]
        x = self.patch_act(x)
        x = self.patch_dropout(x)
        # conv mixing
        x = self.patch_conv(x)               # [N, patch_num, ?]
        x = self.patch_conv_act(x)
        # flatten and project
        x = x.flatten(start_dim=1)           # [N, patch_num * patch_len]
        x = self.patch_proj(x)               # [N, d_model]
        return x

    def _linear_encode(self, x):
        # x: [N, I]
        skip = self.lin_res_proj(x)          # [N, d_model]
        x = self.lin_fc1(x)                  # [N, dim_feedforward]
        x = self.lin_act1(x)
        x = self.lin_ln1(x)
        x = self.lin_dropout(x)
        x = self.lin_fc2(x)                  # [N, d_model]
        x = x + skip
        return x

    def forward(self, s, t, c=None, r=None):
        # Inputs: [B, seq_len, C]
        B, I, C = s.shape
        assert I == self.seq_len, "input seq len mismatch"

        # reshape per-channel as batch dimension for encoders: [B*C, I]
        s_in = s.reshape(B * C, I)
        t_in = t.reshape(B * C, I)
        if c is not None:
            c_in = c.reshape(B * C, I)
        else:
            c_in = None
        if r is not None:
            r_in = r.reshape(B * C, I)
        else:
            r_in = None

        # optional seq preprocessing (TCN)
        if self.tcn is not None:
            s_in = self._seq_preprocess(s_in)
            t_in = self._seq_preprocess(t_in)
            if c_in is not None:
                c_in = self._seq_preprocess(c_in)
            if r_in is not None:
                r_in = self._seq_preprocess(r_in)

        # encode non-linear (s and optional c)
        s_enc = self._nonlinear_encode(s_in)           # [N, d_model]
        if c_in is not None:
            c_enc = self._nonlinear_encode(c_in)
            if self.merge_sc == 'add':
                s_enc = s_enc + c_enc
            else:  # concat
                s_enc = self.merge_sc_proj(torch.cat([s_enc, c_enc], dim=1))

        # encode linear (t and optional r)
        t_enc = self._linear_encode(t_in)              # [N, d_model]
        if r_in is not None:
            r_enc = self._linear_encode(r_in)
            if self.merge_tr == 'add':
                t_enc = t_enc + r_enc
            else:
                t_enc = self.merge_tr_proj(torch.cat([t_enc, r_enc], dim=1))

        # combine encodings -> memory per channel
        # simple additive combine then project
        enc = s_enc + t_enc
        enc = self.combine_proj(enc)                   # [N, d_model]

        # Transformer expects shapes: memory: (S, N, E), tgt: (T, N, E)
        # We'll use memory length S = 1 (one token per channel), batch N = B*C
        memory = enc.unsqueeze(0)                      # [1, N, d_model]

        # prepare target queries: (pred_len, N, d_model)
        tgt = self.tgt_queries.unsqueeze(1).expand(-1, B * C, -1).contiguous()  # [pred_len, N, d_model]

        # decode
        dec_out = self.transformer_decoder(tgt=tgt, memory=memory)  # [pred_len, N, d_model]

        # project to scalar and reshape to [B, pred_len, C]
        out = self.head(dec_out)                       # [pred_len, N, 1]
        out = out.squeeze(-1)                          # [pred_len, N]
        out = out.permute(1, 0)                        # [N, pred_len]
        out = out.reshape(B, C, self.pred_len)         # [B, C, pred_len]
        out = out.permute(0, 2, 1)                     # [B, pred_len, C]
        return out
# ...existing code...