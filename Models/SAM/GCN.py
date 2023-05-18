import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

def cal_adj_norm(adj):
    node_num = adj.shape[0]
    adj = np.asarray(adj)
    adj_ = adj + np.eye(node_num)
    # 度矩阵
    d = np.sum(adj_,1)
    d_sqrt = np.power(d, -0.5)
    d_sqrt = np.diag(d_sqrt)
    adj_norm = np.dot(np.dot(d_sqrt, adj_), d_sqrt)
    return adj_norm

class Graph_convolution(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False, activation='relu'):
        super(Graph_convolution, self).__init__()
        self.bias = None
        # 权重矩阵
        self.weight = nn.Parameter(torch.randn(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        # 激活函数
        if activation == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.Sigmoid()

    def forward(self, x, adj):
        x = x.reshape(69, -1)
        x = torch.tensor(x, dtype=torch.float)
        adj = torch.tensor(adj, dtype=torch.float)
        out = torch.mm(adj, x)
        out = torch.tensor(out, dtype=torch.float)
        out = torch.mm(out, self.weight)
        if self.bias:
            out += self.bias
        return self.act(out)


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim1, output_dim2, adj):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.adj = adj
        self.layer1 = Graph_convolution(self.input_dim, self.output_dim1)
        self.layer2 = Graph_convolution(self.output_dim1, self.output_dim2, activation='sigmoid')

    def forward(self, x):
        self.adj = cal_adj_norm(self.adj)
        B = []
        for i in range(x.shape[1]):
            A = []
            for j in range(x.shape[2]):
                out = self.layer1(x[:, i, j, :], self.adj)
                out = self.layer2(out, self.adj)
                A.append(out)
            A = torch.stack(A)
            B.append(A)
        B = torch.stack(B)
        return B
