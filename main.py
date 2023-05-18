import numpy as np
import pandas as pd
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from Models.MSTSN import MSTSN
from util.dataload import MyDataset, read_data, load_adj
from util.framework import Framework
from util.log import getLogger
from util.params import parse_args

if __name__ == "__main__":
    config = parse_args()
    logger = getLogger(name="model_log.txt", config=config)

    config.X_train, config.Y_train, config.X_dev, config.Y_dev, config.X_test,\
                                                    config.Y_test = read_data(config.data_source)
    config.adj = load_adj(config.adj_path)
    config.gcn_inputdim = config.X_train.shape[-1]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.device = device

    model = MSTSN(config)
    framework = Framework(config, model, logger)

    if config.do_train:
        framework.train()
    if config.do_test:
        framework.load_model(f"{config.output_pth}")
