import torch
from torch import nn
import torch.nn.functional as F

class SSMConv1D(nn.Module):
    def __init__(self, channels, kernel_size=7, rank=1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.rank = rank

        # A: [channels, rank]
        self.A = nn.Parameter(torch.randn(channels, rank))
        # B: [channels, rank]
        self.B = nn.Parameter(torch.randn(channels, rank))
        # c: [channels, rank]
        self.C = nn.Parameter(torch.randn(channels, rank))

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        kernels = []
        for ch in range(self.channels):
            # F_k = c^T A^k B, k = 0..kernel_size-1
            Ak = torch.eye(self.rank, device=x.device)
            Fk = []
            for k in range(self.kernel_size):
                val = self.C[ch] @ Ak @ self.B[ch]
                Fk.append(val)
                Ak = Ak @ torch.diag(self.A[ch])
            kernel = torch.stack(Fk)  # [kernel_size]
            kernels.append(kernel)
        # kernels: [channels, kernel_size]
        kernels = torch.stack(kernels)  # [C, kernel_size]

        # Depthwise conv1d
        x = x.permute(0, 2, 1)  # [B, C, T]
        out = []
        for ch in range(self.channels):
            out_ch = F.conv1d(
                x[:, ch:ch+1, :],
                kernels[ch:ch+1, :].unsqueeze(1),  # [out_ch, in_ch=1, k]
                padding=self.kernel_size // 2
            )
            out.append(out_ch)
        out = torch.cat(out, dim=1)  # [B, C, T]
        out = out.permute(0, 2, 1)  # [B, T, C]
        return out

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, c_in):
        super(Network, self).__init__()

        # Parameters
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in  = c_in
        self.period_len = 24
        self.d_model = 128

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False
        )
        self.pool = nn.AvgPool1d(
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.seg_num_y)
        )

        self.ssm = SSMConv1D(self.enc_in, kernel_size=7, rank=1)

        # Linear Stream
        self.fc5 = nn.Linear(seq_len, pred_len * 2)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        s = s.permute(0,2,1) # [Batch, Channel, Input]
        t = t.permute(0,2,1) # [Batch, Channel, Input]

        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        t = torch.reshape(t, (B*C, I))

        # Seasonal Stream: Conv1d + Pooling
        s_conv = self.conv1d(s.reshape(-1, 1, self.seq_len))
        s_pool = self.pool(s.reshape(-1, 1, self.seq_len))
        s_concat = s_conv + s_pool
        s_concat = s_concat.reshape(-1, self.enc_in, self.seq_len) + s
        s = s_concat.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        y = self.mlp(s)
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1) # [B, pred_len, enc_in]

        # Thêm SSMConv1D để học động lực chuỗi thời gian
        y = self.ssm(y)  # [B, pred_len, enc_in]

        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [Batch, Output, Channel] = [B, pred_len, C]

        return t + y