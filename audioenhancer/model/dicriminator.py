from torch import nn

import math
from audioenhancer.constants import MAX_AUDIO_LENGTH


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, last_out_channel=512):
        super(MLP, self).__init__()

        self.up_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.gate = nn.Linear(in_features, hidden_features, bias=False)
        reduction = int(math.log2(last_out_channel))
        linears = []
        for i in range(reduction):
            linears.append(nn.Linear(hidden_features*2, hidden_features, bias=False))
        self.linears = nn.ModuleList(linears)

    def forward(self, x):
        gate = self.gate(x)
        h = self.up_proj(x) * nn.ReLU()(gate)
        for linear in self.linears:
            bsz, seq_len, dim = h.shape
            h = h.view(bsz, seq_len//2, dim*2, )
            h = linear(h)
        return h

def calculate_output_size(input_size, kernel_size, stride, padding):
    return int(((input_size - kernel_size + 2*padding) / stride)) + 1

class Discriminator(nn.Module):
    def __init__(self, in_channels=2, kernel_size=3, stride=2, padding=1, sample_rate=16_000):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=kernel_size, stride=stride, padding=padding)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=kernel_size, stride=stride, padding=padding)

        input_size = sample_rate * MAX_AUDIO_LENGTH
        for _ in range(4):
            input_size = calculate_output_size(input_size, kernel_size, stride, padding)
        self.mlp = MLP(input_size, 2048, last_out_channel=512)

        self.head = nn.Linear(2048, 1, bias=False)

        self.act_fn = nn.LeakyReLU(0.2)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, 0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.03)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + self.act_fn(self.conv2(x))
        x = self.conv3(x)
        x = self.conv4(x) + self.act_fn(self.conv4(x))
        x = self.mlp(x).squeeze(1)
        x = self.head(x)
        return x
