import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    raise ValueError(f"{activation} is not available. Use 'relu', 'gelu', or a callable.")


class FlattenHead(nn.Module):
    def __init__(self, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class SublayerConnection(nn.Module):
    def __init__(self, enable_res_parameter: bool, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.enable = enable_res_parameter
        if enable_res_parameter:
            self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x: Tensor, out_x: Tensor) -> Tensor:
        if not self.enable:
            return x + self.dropout(out_x)
        else:
            return x + self.dropout(self.a * out_x)


class ConvEncoderLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 256, kernel_size: int = 9, dropout: float = 0.1,
                 activation: str = "gelu", enable_res_param: bool = True, norm: str = 'batch',
                 re_param: bool = True, small_ks: int = 3):
        super().__init__()
        self.norm_tp = norm
        self.re_param = re_param
        if not re_param:
            self.DW_conv = nn.Conv1d(d_model, d_model, kernel_size, 1, 'same', groups=d_model)
        else:
            self.large_ks = kernel_size
            self.small_ks = small_ks
            self.DW_conv_large = nn.Conv1d(d_model, d_model, kernel_size, stride=1, padding='same', groups=d_model)
            self.DW_conv_small = nn.Conv1d(d_model, d_model, small_ks, stride=1, padding='same', groups=d_model)
            self.DW_infer = nn.Conv1d(d_model, d_model, kernel_size, stride=1, padding='same', groups=d_model)
        self.dw_act = get_activation_fn(activation)
        self.sublayerconnect1 = SublayerConnection(enable_res_param, dropout)
        self.dw_norm = nn.BatchNorm1d(d_model) if norm == 'batch' else nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Conv1d(d_model, d_ff, 1, 1),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Conv1d(d_ff, d_model, 1, 1),
        )
        self.sublayerconnect2 = SublayerConnection(enable_res_param, dropout)
        self.norm_ffn = nn.BatchNorm1d(d_model) if norm == 'batch' else nn.LayerNorm(d_model)

    def _get_merged_param(self):
        left_pad = (self.large_ks - self.small_ks) // 2
        right_pad = (self.large_ks - self.small_ks) - left_pad
        module_output = copy.deepcopy(self.DW_conv_large)
        module_output.weight = torch.nn.Parameter(
            module_output.weight + F.pad(self.DW_conv_small.weight, (left_pad, right_pad), value=0)
        )
        module_output.bias = torch.nn.Parameter(module_output.bias + self.DW_conv_small.bias)
        self.DW_infer = module_output

    def forward(self, src: Tensor) -> Tensor:  # [B, C=d_model, L=patch_num]
        if not self.re_param:
            out_x = self.DW_conv(src)
        else:
            if self.training:
                large_out, small_out = self.DW_conv_large(src), self.DW_conv_small(src)
                out_x = large_out + small_out
            else:
                self._get_merged_param()
                out_x = self.DW_infer(src)
        src2 = self.dw_act(out_x)
        src = self.sublayerconnect1(src, src2)
        src = src.permute(0, 2, 1) if self.norm_tp != 'batch' else src
        src = self.dw_norm(src)
        src = src.permute(0, 2, 1) if self.norm_tp != 'batch' else src
        src2 = self.ff(src)
        src2 = self.sublayerconnect2(src, src2)
        src2 = src2.permute(0, 2, 1) if self.norm_tp != 'batch' else src2
        src2 = self.norm_ffn(src2)
        src2 = src2.permute(0, 2, 1) if self.norm_tp != 'batch' else src2
        return src2


class ConvEncoder(nn.Module):
    def __init__(self, kernel_sizes, d_model, d_ff=None, norm='batch', dropout=0., activation='gelu',
                 enable_res_param=True, n_layers=3, re_param=True, re_param_kernel=3):
        super().__init__()
        self.layers = nn.ModuleList([
            ConvEncoderLayer(d_model, d_ff=d_ff, kernel_size=kernel_sizes[i], dropout=dropout,
                             activation=activation, enable_res_param=enable_res_param, norm=norm,
                             re_param=re_param, small_ks=re_param_kernel)
            for i in range(n_layers)
        ])

    def forward(self, src: Tensor) -> Tensor:
        output = src
        for mod in self.layers:
            output = mod(output)
        return output


class ConviEncoder(nn.Module):
    def __init__(self, patch_num, patch_len, kernel_sizes=(11, 15, 21, 29, 39, 51), n_layers=6,
                 d_model=128, d_ff=256, norm='batch', dropout=0., act="gelu",
                 enable_res_param=True, re_param=True, re_param_kernel=3):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.W_P = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = ConvEncoder(kernel_sizes, d_model, d_ff=d_ff, norm=norm, dropout=dropout,
                                   activation=act, enable_res_param=enable_res_param, n_layers=n_layers,
                                   re_param=re_param, re_param_kernel=re_param_kernel)

    def forward(self, x: Tensor) -> Tensor:  # x: [bs, nvars, patch_len, patch_num]
        n_vars = x.shape[1]
        x = x.permute(0, 1, 3, 2)             # [bs, nvars, patch_num, patch_len]
        x = self.W_P(x)                        # [bs, nvars, patch_num, d_model]
        u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # [bs*nvars, patch_num, d_model]
        z = self.encoder(u.permute(0, 2, 1)).permute(0, 2, 1)                    # [bs*nvars, patch_num, d_model]
        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))             # [bs, nvars, patch_num, d_model]
        z = z.permute(0, 1, 3, 2)                                                # [bs, nvars, d_model, patch_num]
        return z


class ConvBackbone(nn.Module):
    """ConvTimeNet-style backbone adapted to produce [B, C, pred_len] from [B, C, seq_len]."""
    def __init__(self, c_in: int, seq_len: int, pred_len: int, patch_len: int, stride: int,
                 n_layers: int = 4, dw_ks=(9, 11, 15, 21, 29, 39), d_model: int = 64, d_ff: int = 256,
                 norm: str = 'batch', dropout: float = 0.1, act: str = 'gelu',
                 enable_res_param: bool = True, re_param: bool = True, re_param_kernel: int = 3,
                 head_dropout: float = 0.1, padding_patch: str | None = None):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.n_vars = c_in
        patch_num = int((seq_len - patch_len) / stride + 1)
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1
        self.head_nf = d_model * patch_num
        self.backbone = ConviEncoder(patch_num=patch_num, patch_len=patch_len, kernel_sizes=dw_ks,
                                     n_layers=n_layers, d_model=d_model, d_ff=d_ff, norm=norm,
                                     dropout=dropout, act=act, enable_res_param=enable_res_param,
                                     re_param=re_param, re_param_kernel=re_param_kernel)
        self.head = FlattenHead(self.n_vars, self.head_nf, pred_len, head_dropout=head_dropout)

    def forward(self, z: Tensor) -> Tensor:  # z: [B, C, L]
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)   # [B, C, patch_num, patch_len]
        z = z.permute(0, 1, 3, 2)                                           # [B, C, patch_len, patch_num]
        z = self.backbone(z)                                                # [B, C, d_model, patch_num]
        z = self.head(z)                                                    # [B, C, pred_len]
        return z
