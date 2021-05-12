from parameters import *
import torch as th
from torch import nn
import torch.optim as optim

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, output_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        hidden = th.cat((hidden[-2, :, :], hidden[-1, :, :]), dim = 1)
        out = self.linear(hidden)
        return out