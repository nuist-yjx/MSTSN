import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn


def create_dataset(dataset, look_back=6):
    datax, datay = [], []
    for i in range(len(dataset) - look_back + 1):
        a = dataset[i:(i + look_back), :-1]
        datax.append(a)
        datay.append(dataset[i + look_back - 1, -1])
    return np.array(datax), np.array(datay)


def create_dataset2(dataset, look_back=5):
    datax, datay = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :-1]
        datax.append(a)
        datay.append(dataset[i + look_back, -1])
    return np.array(datax), np.array(datay)


def read_data(path):
    all_data = []
    all_datasets = []
    # all_X = []
    # all_Y = []
    all_X_train = []
    all_Y_train = []
    all_X_dev = []
    all_Y_dev = []
    all_X_test = []
    all_Y_test = []
    for i in range(1, 70):
        d = pd.read_csv(f'{path}/{i}.csv', usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13])
        dataset = d.values
        X, Y = create_dataset(dataset, look_back=6)
        X = X.reshape(-1, 60)
        Y = Y.reshape(-1, 1)
        Z = np.concatenate([X, Y], axis=1)
        X, Y = create_dataset2(Z)
        X_train = X[:-61]
        Y_train = Y[:-61]
        X_dev = X[-61:-48]
        Y_dev = Y[-61:-48]
        X_test = X[-48:]
        Y_test = Y[-48:]
        all_data.append(d)
        all_datasets.append(dataset)
        all_X_train.append(X_train)
        all_Y_train.append(Y_train)
        all_X_dev.append(X_dev)
        all_Y_dev.append(Y_dev)
        all_X_test.append(X_test)
        all_Y_test.append(Y_test)

    all_X_train = np.array(all_X_train)
    all_Y_train = np.array(all_Y_train)
    all_X_train = all_X_train.reshape(-1, 5, 60)
    all_Y_train = all_Y_train.reshape(-1, 1)

    all_X_dev = np.array(all_X_dev)
    all_Y_dev = np.array(all_Y_dev)
    all_X_dev = all_X_dev.reshape(-1, 5, 60)
    all_Y_dev = all_Y_dev.reshape(-1, 1)

    all_X_test = np.array(all_X_test)
    all_Y_test = np.array(all_Y_test)
    all_X_test = all_X_test.reshape(-1, 5, 60)
    all_Y_test = all_Y_test.reshape(-1, 1)

    all_X_train = all_X_train.reshape(all_X_train.shape[0], -1)
    all_Y_train = all_Y_train.reshape(all_Y_train.shape[0], -1)
    all_X_dev = all_X_dev.reshape(all_X_dev.shape[0], -1)
    all_Y_dev = all_Y_dev.reshape(all_Y_dev.shape[0], -1)
    all_X_test = all_X_test.reshape(all_X_test.shape[0], -1)
    all_Y_test = all_Y_test.reshape(all_Y_test.shape[0], -1)
    print('X_train shape: ', all_X_train.shape)
    print('X_dev shape: ', all_X_dev.shape)
    print('X_test shape: ', all_X_test.shape)

    data_scaler = preprocessing.MinMaxScaler()
    target_scaler = preprocessing.MinMaxScaler()
    X = np.concatenate([all_X_train, all_X_dev, all_X_test], axis=0)
    # Y = np.concatenate([Y_train,Y_test], axis=0)
    X = data_scaler.fit_transform(X)
    # Y = target_scaler.fit_transform(Y)

    # #reshape
    all_X_train = X[:43125]
    all_X_dev = X[43125:43125 + 897]
    all_X_test = X[44022:]

    X_train1 = all_X_train.reshape(-1, 5, 60)
    Y_train1 = all_Y_train.reshape(-1, 1, 1)

    X_dev1 = all_X_dev.reshape(-1, 5, 60)
    Y_dev1 = all_Y_dev.reshape(-1, 1, 1)
    # #获取test的值
    X_test1 = all_X_test.reshape(-1, 5, 60)
    Y_test1 = all_Y_test.reshape(-1, 1, 1)

    X_train2 = all_X_train.reshape(-1, 69, 5, 60)
    Y_train2 = all_Y_train.reshape(-1, 69, 1)

    X_dev2 = all_X_dev.reshape(-1, 69, 5, 60)
    Y_dev2 = all_Y_dev.reshape(-1, 69, 1)

    # #获取test的值
    X_test2 = all_X_test.reshape(-1, 69, 5, 60)
    Y_test2 = all_Y_test.reshape(-1, 69, 1)
    return X_train2, Y_train2, X_dev2, Y_dev2, X_test2, Y_test2


def load_adj(path):
    adj_matrix = pd.read_csv(path, header=None)
    adj = np.mat(adj_matrix)
    return adj


class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.all_X = X
        self.all_Y = Y

    def __getitem__(self, idx):

        X = self.all_X[idx,:,:,:]
        Y = self.all_Y[idx,:,:]
        return X, Y

    def __len__(self):
        return self.all_X.__len__()

    def collate_fn(self, batch):
        batch_input = []
        batch_label = []
        for input, label in batch:
            for i in range(len(label)):
                lab = label[i]
                if lab > -0.5:
                    label[i] = 0
                elif lab <= -0.5 and lab >= -1.0:
                    label[i] = 1
                elif lab < -1.0 and lab >= -1.5:
                    label[i] = 2
                elif lab < -1.5 and lab >= -2.0:
                    label[i] = 3
                else:
                    label[i] = 4

            batch_input.append(input)
            batch_label.append(label)

        batch_input = torch.tensor(batch_input, dtype=float).transpose(0,1)
        batch_label = torch.tensor(batch_label, dtype=int).transpose(0,1)

        return batch_input, batch_label

