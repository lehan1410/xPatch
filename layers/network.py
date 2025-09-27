import torch
from torch import nn

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size-1)*dilation//2, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.GELU()
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        res = self.residual(x)
        return out + res

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, c_in, cycle_len):
        super(Network, self).__init__()

        self.pred_len = pred_len
        self.seq_len = seq_len
        self.enc_in  = c_in
        self.period_len = 24
        self.d_model = 128
        self.cycle_len = cycle_len

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.temporalQuery = nn.Parameter(torch.zeros(24, self.enc_in), requires_grad=True)

        # TCN thay cho Conv1d
        self.tcn = nn.Sequential(
            TCNBlock(1, 1, kernel_size=3, dilation=1),
            TCNBlock(1, 1, kernel_size=3, dilation=2)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.seg_num_y)
        )

        # Linear Stream
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.gelu1 = nn.GELU()
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t, cycle_index):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        s = s.permute(0,2,1) # [Batch, Channel, Input]
        t = t.permute(0,2,1) # [Batch, Channel, Input]

        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        t = torch.reshape(t, (B*C, I))

        gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=s.device).view(1, -1)) % self.cycle_len
        temporal_emb = self.temporalQuery[gather_index]  # [B, seq_len, C]
        temporal_emb = temporal_emb.permute(0, 2, 1)     # [B, C, seq_len]
        s_cat = torch.cat([s, temporal_emb], dim=1)      # [B, 2*C, seq_len]

        # TCN thay cho Conv1d
        s = self.tcn(s_cat) + s  # [B, C, seq_len]

        s = s.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        y = self.mlp(s)
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1)

        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.avgpool1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [Batch, Output, Channel] = [B, pred_len, C]

        return t + y