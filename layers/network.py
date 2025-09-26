import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, enc_in, d_model=128):
        super(Network, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.enc_in = enc_in
        self.d_model = d_model

        # Khởi tạo với giá trị mặc định, sẽ cập nhật lại trong forward
        self.seg_num_x = self.seq_len // 24
        self.seg_num_y = self.pred_len // 24
        self.period_len = 24

        # Khởi tạo conv1d và mlp với giá trị mặc định
        self.conv1d = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2, padding_mode="zeros", bias=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.seg_num_y)
        )

    @staticmethod
    def get_optimal_M(s, fs=1.0, energy_ratio=0.99):
        # s: [Batch, Channel, Input]
        s_flat = s.mean(dim=1)  # [Batch, Input]
        S_fft = torch.fft.fft(s_flat)
        mag = torch.abs(S_fft)
        energy = mag**2
        energy_cumsum = torch.cumsum(energy, dim=-1)
        total_energy = energy_cumsum[:, -1]
        idx = (energy_cumsum >= energy_ratio * total_energy.unsqueeze(-1)).float().argmax(dim=-1)
        f_max = idx * fs / s.shape[-1]
        f_max_energy = f_max.max().item()
        M = max(1, int(fs / (2 * f_max_energy)))
        return M, f_max_energy

    def forward(self, s, t):
        # s: [Batch, Input, Channel] (seasonal)
        # t: [Batch, Input, Channel] (trend)

        batch_size = s.shape[0]
        s = s.permute(0,2,1) # [Batch, Channel, Input]
        t = t.permute(0,2,1) # [Batch, Channel, Input]

        # Chọn M tối ưu cho seasonal
        M, fmax = self.get_optimal_M(s)
        self.seg_num_x = self.seq_len // M
        self.seg_num_y = self.pred_len // M
        self.period_len = M

        # Khởi tạo lại conv1d và mlp với M mới
        self.conv1d = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=1 + 2 * (M // 2),
            stride=1, padding=M // 2, padding_mode="zeros", bias=False
        ).to(s.device)
        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.seg_num_y)
        ).to(s.device)

        x = s  # [batch, enc_in, seq_len]

        # 1D convolution aggregation
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        # Downsampling: b,c,s -> bc,n,w -> bc,w,n
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        # Sparse forecasting qua MLP
        y = self.mlp(x)
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

        # Đảm bảo t có shape [batch, enc_in, seq_len] trước khi slice
        if t.shape[2] != self.seq_len:
            t = t.permute(0, 2, 1)  # [batch, enc_in, seq_len]
        trend_part = t[:, :, -self.pred_len:]  # [batch, enc_in, pred_len]
        y = y + trend_part

        return y