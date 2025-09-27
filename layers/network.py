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

        self.depthwise = nn.Conv1d(
            in_channels=self.enc_in, out_channels=self.enc_in,
            kernel_size=3, padding=1, groups=self.enc_in, bias=False
        )
        self.pointwise = nn.Conv1d(
            in_channels=self.enc_in, out_channels=self.enc_in,
            kernel_size=1, bias=False
        )
        self.dw_act = nn.GELU()

        self.cycle_len = self.seq_len
        self.temporalQuery = torch.nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True)

        # MLP cho seasonal stream với LayerNorm
        self.mlp = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.seg_num_y),
            nn.LayerNorm(self.seg_num_y)
        )

        # Linear Stream
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.gelu1 = nn.GELU()
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc7 = nn.Linear(pred_len * 2, pred_len)
        self.ln2 = nn.LayerNorm(pred_len)
        self.fc8 = nn.Linear(pred_len, pred_len)
        self.ln3 = nn.LayerNorm(pred_len)

    def forward(self, s, t):
        # s: [Batch, Input, Channel]
        # t: [Batch, Input, Channel]
        s = s.permute(0,2,1) # [Batch, Channel, Input]
        t = t.permute(0,2,1) # [Batch, Channel, Input]

        B = s.shape[0]
        C = s.shape[1]
        I = s.shape[2]
        t = torch.reshape(t, (B*C, I))

        s = s + self.temporalQuery[:self.seq_len, :].T.unsqueeze(0)

        s = self.conv1d(s.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + s
        s_dw = self.depthwise(s)
        s_pw = self.pointwise(self.dw_act(s_dw))
        s = s_pw.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        y = self.mlp(s)
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1)

        # Linear Stream
        t = self.fc5(t)
        t = t.unsqueeze(1)             # [B*C, 1, pred_len*4]
        t = self.avgpool1(t)           # [B*C, 1, pred_len*2]
        t = t.squeeze(1)
        t = self.gelu1(t)
        t = self.ln1(t)
        t = self.fc7(t)
        t = self.ln2(t)
        t = self.fc8(t)
        t = self.ln3(t)
        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1) # [Batch, Output, Channel] = [B, pred_len, C]

        return t + y