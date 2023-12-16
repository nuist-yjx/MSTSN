from torch.nn import Flatten

from Models.SAM.GCN import GCN
from Models.TEM.GRU import GRU


import torch.nn as nn


class MSTSN(nn.Module):
    def __init__(self, config):
        super(MSTSN, self).__init__()
        self.config = config
        self.gcn_dim1 = config.gcn_dim1
        self.gcn_dim2 = config.gcn_dim2
        self.gcn = GCN(config.gcn_inputdim, self.gcn_dim1, self.gcn_dim2, config.adj)
        self.gru = GRU(config.gcn_dim2 - config.kernel_size + 1, config.gru_dim, config.gru_num, config.gru_output_dim)
        self.atten = nn.MultiheadAttention(config.gru_output_dim, num_heads=1, dropout=0.2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = Flatten()
        self.linear = nn.Linear(config.gru_output_dim, config.label_num)
        self.lossfn = nn.CrossEntropyLoss()

    def forward(self, input, label, istrain=False):
        gcn_out = self.gcn(input)
        gcn_out = gcn_out.reshape(gcn_out.shape[0] * gcn_out.shape[2], gcn_out.shape[1], gcn_out.shape[3])
        gru_out = self.gru(gcn_out)
        att_out, _ = self.atten(gru_out, gru_out, gcn_out)
        output = self.flatten(self.gap(att_out.transpose(1, 2)))
        output = self.linear(output)
        if istrain:
            label = label.reshape(-1, 1)
            loss = self.lossfn(output, label.squeeze(1))
            return loss
        else:
            return output
