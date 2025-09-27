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
        self.pool = nn.AvgPool1d(
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1,
            padding=self.period_len // 2
        )

        self.mlp_time = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.seg_num_y)
        )
        # MLP cho tương tác channel
        self.mlp_channel = nn.Sequential(
            nn.Linear(self.period_len, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.period_len)
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

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.gelu2 = nn.GELU()
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)
        self.gelu3 = nn.GELU()

        # Streams Concatination
        self.fc8 = nn.Linear(pred_len * 2, pred_len)

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

        y_time = self.mlp_time(s)
        s_channel = s.permute(0, 2, 1)  # [*, seg_num_x, period_len]
        y_channel = self.mlp_channel(s_channel)  # [*, seg_num_x, period_len]
        y_channel = y_channel.permute(0, 2, 1)  # [*, period_len, seg_num_x]
        # Resize y_channel để match shape với y_time nếu cần
        if y_channel.shape[-1] != y_time.shape[-1]:
            y_channel = y_channel[..., :y_time.shape[-1]]

        s = y_time + y_channel  
        
        y = self.mlp(s)
        y = y.permute(0, 2, 1).reshape(B, self.enc_in, self.pred_len)
        y = y.permute(0, 2, 1) # [B, pred_len, enc_in]


        # Linear Stream
        t = self.fc5(t)
        t = self.gelu1(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.gelu2(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)
        t = self.gelu3(t)

        t = self.fc8(t)

        t = torch.reshape(t, (B, C, self.pred_len))
        t = t.permute(0,2,1)

        return t + y