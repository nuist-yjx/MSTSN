import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class TC(nn.Module):
    def __init__(self, num_features,
                 conv_nf, kernel_size,fc_drop_p=0.3):
        super(TC, self).__init__()

        self.kernel_size = kernel_size
        self.conv_nf = conv_nf
        self.num_features = num_features
        self.fc_drop_p = fc_drop_p


        self.conv = nn.Conv1d(self.num_features, self.conv_nf, kernel_size=self.kernel_size)
        self.bn = nn.BatchNorm1d(self.conv_nf)
        self.se = SELayer(self.conv_nf)  # ex 128
        self.relu = nn.ReLU()
        self.convDrop = nn.Dropout(self.fc_drop_p)

    def forward(self, x):
        ''' input x should be in size [B,T,F], where
            B = Batch size
            T = Time samples
            F = features
        '''
        x = self.convDrop(self.relu(self.bn(self.conv(x))))
        x = self.se(x)
        return x