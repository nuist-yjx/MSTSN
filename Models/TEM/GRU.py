# 定义GRU网络
from torch import nn


class GRU(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0 = hidden
        output, h_0 = self.gru(x, h_0)
        batch_size, timestep, hidden_size = output.shape
        output = output.reshape(-1, hidden_size)
        output = self.fc(output)
        output = output.reshape(batch_size, timestep,  -1)
        return output
