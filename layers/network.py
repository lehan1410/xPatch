import torch
from torch import nn

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
        self.bn_s = nn.BatchNorm1d(self.enc_in)

        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.seg_num_y)
        )
        self.bn_mlp = nn.BatchNorm1d(self.pred_len)  # Sửa lại cho đúng shape

        # Linear Stream (trend)
        self.fc5 = nn.Linear(seq_len, pred_len * 2)
        self.gelu1 = nn.GELU()
        self.bn_t1 = nn.BatchNorm1d(c_in)
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        self.bn_t2 = nn.BatchNorm1d(c_in)
        self.fc8 = nn.Linear(pred_len, pred_len)

    def forward(self, s, t):
        # s, t: [Batch, Input, Channel]
        s = s.permute(0,2,1) # [Batch, Channel, Input]
        t = t.permute(0,2,1) # [Batch, Channel, Input]

        B, C, I = s.shape

        # Seasonal Stream
        s_res = s.clone()
        s = self.conv1d(s.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + s_res
        s = self.bn_s(s)
        s = s.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        y = self.mlp(s)
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1)  # [B, pred_len, enc_in]
        y = self.bn_mlp(y)      # BatchNorm1d(pred_len)
        y = y.permute(0, 2, 1)  # [B, enc_in, pred_len]

        # Trend Stream
        t = torch.reshape(t, (B*C, I))
        t = self.fc5(t)
        t = self.gelu1(t)
        t = t.reshape(B, C, -1)
        t = self.bn_t1(t)
        t = t.reshape(B*C, -1)
        t = self.ln1(t)
        t = self.fc7(t)
        t = t.reshape(B, C, -1)
        t = self.bn_t2(t)
        t = t.reshape(B*C, -1)
        t = self.fc8(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [Batch, Output, Channel]

        return t + y