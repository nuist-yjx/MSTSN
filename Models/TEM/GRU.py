# 定义GRU网络
from torch import nn


class GRU(nn.Module):
    def __init__(self, feature_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size  # 隐层大小
        self.num_layers = num_layers  # gru层数
        # feature_size为特征维度，就是每个时间点对应的特征数量，这里为1
        self.gru = nn.GRU(feature_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]  # 获取批次大小

        # 初始化隐层状态
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0 = hidden

        # GRU运算
        output, h_0 = self.gru(x, h_0)

        # 获取GRU输出的维度信息
        batch_size, timestep, hidden_size = output.shape

        # 将output变成 batch_size * timestep, hidden_dim
        output = output.reshape(-1, hidden_size)

        # 全连接层
        output = self.fc(output)  # 形状为batch_size * timestep, 1

        # 转换维度，用于输出
        output = output.reshape(batch_size,timestep,  -1)

        # 我们只需要返回最后一个时间片的数据即可
        return output
