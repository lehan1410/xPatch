import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, enc_in, period_len=24, d_model=128, model_type='mlp'):
        super(Network, self).__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.period_len = period_len
        self.d_model = d_model
        self.model_type = model_type

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(
            in_channels=1, out_channels=1,
            kernel_size=1 + 2 * (self.period_len // 2),
            stride=1, padding=self.period_len // 2,
            padding_mode="zeros", bias=False
        )

        if self.model_type == 'linear':
            self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        elif self.model_type == 'mlp':
            self.mlp = nn.Sequential(
                nn.Linear(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.seg_num_y)
            )

    def forward(self, s, t):
        # s: [Batch, Input, Channel] (seasonal đã tách)
        # t: [Batch, Input, Channel] (trend đã tách)

        batch_size = s.shape[0]
        s = s.permute(0, 2, 1)  # [Batch, Channel, Input]
        t = t.permute(0, 2, 1)  # [Batch, Channel, Input]

        # Xử lý trực tiếp seasonal, KHÔNG cần trừ trend
        seq_mean = torch.mean(s, dim=1).unsqueeze(1)  # [Batch, 1, Input]
        x = (s - seq_mean)  # [Batch, Channel, Input]
        x = x.reshape(-1, 1, self.seq_len)  # [Batch*Channel, 1, seq_len]
        conv_out = self.conv1d(x)  # [Batch*Channel, 1, seq_len]
        x = conv_out + x  # [Batch*Channel, 1, seq_len]
        x = x.reshape(batch_size, self.enc_in, self.seq_len)
        x = x.unfold(dimension=2, size=self.period_len, step=self.period_len)  # [batch, enc_in, seg_num_x, period_len]
        x = x.mean(dim=-1)

        # sparse forecasting
        if self.model_type == 'linear':
            y = self.linear(x)
        elif self.model_type == 'mlp':
            y = self.mlp(x)

        # upsampling: bc,w,m -> bc,m,w -> b,c,s
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)

        # cộng lại trend để khôi phục chuỗi gốc
        trend_part = t[:, :, -self.pred_len:]  # [Batch, Channel, pred_len]
        y = y.permute(0, 2, 1) + trend_part  # [Batch, pred_len, Channel]

        return y