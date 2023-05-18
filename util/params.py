import random
import argparse
import torch
import os
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options
    parser.add_argument("--data_source", type=str, default="./datasets/data", help="Source of the dataset")
    parser.add_argument("--adj_path", type=str, default="./datasets/Ajuzheng.csv")
    parser.add_argument("--output_log_path", type=str, default="./output/log")
    parser.add_argument("--output_pth", type=str, default="./output/saved_pth/MSTSN_model.pth")

    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument("--do_test", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--gcn_dim1", default=64)
    parser.add_argument("--gcn_dim2", default=128)
    parser.add_argument("--conv_nf", default=128)
    parser.add_argument("--inchannel",default=5)
    parser.add_argument("--kernel_size", default=8)
    parser.add_argument("--gru_dim", default=256)
    parser.add_argument("--gru_num", default=2)
    parser.add_argument("--gru_output_dim", default=128)
    parser.add_argument("--lr", default=0.0001)
    parser.add_argument("--batch_size", default=32)
    parser.add_argument("--epoch", default=100)
    parser.add_argument("--drop_out", default=0.2)
    parser.add_argument("--label_num", default=5)

    args = parser.parse_args()
    seed_everything(args.seed)

    return args
