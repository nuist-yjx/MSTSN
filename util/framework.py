import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

from util.dataload import MyDataset, read_data
from util.params import parse_args
from tqdm import tqdm

class Framework(object):
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.device = config.device

    def set_learning_setting(self, config, model):
        parameters = model.parameters()
        optimizer = torch.optim.Adam(parameters, lr=config.lr)
        return optimizer

    def get_dataloader(self, input, label, config):
        dataset = MyDataset(input, label)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
        return dataloader

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))

    def train(self):
        train_loader = self.get_dataloader(self.config.X_train, self.config.Y_train, self.config)
        dev_loader = self.get_dataloader(self.config.X_dev, self.config.Y_dev, self.config)
        optimizer = self.set_learning_setting(self.config, self.model)
        self.logger.info("Training ·······")
        max_f1 = 0.
        for e in range(self.config.epoch):
            train_loss = 0.0
            train_num = 0
            self.model.train()
            for batch_train_input, batch_train_label in tqdm(train_loader):
                batch_train_label = batch_train_label.to(self.device)
                batch_train_input = batch_train_input.to(self.device)

                loss = self.model(batch_train_input, batch_train_label, istrain=True)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss
                train_num += 1

            all_pred = []
            all_label = []
            self.model.eval()
            for batch_dev_input, batch_dev_label in dev_loader:
                batch_dev_input = batch_dev_input.to(self.device)
                batch_dev_label = batch_dev_label.to(self.device)
                pred = self.model(batch_dev_input, batch_dev_label)
                pred = torch.argmax(pred,dim=-1)
                all_pred.append(pred)
                all_label.append(batch_dev_label)

            all_pred = torch.tensor([item.cpu().detach().numpy() for item in all_pred])
            all_label = torch.tensor([item.cpu().detach().numpy() for item in all_label])
            f1 = f1_score(all_pred.reshape(-1), all_label.reshape(-1), average="weighted")
            acc = accuracy_score(all_pred.reshape(-1), all_label.reshape(-1))
            recall = recall_score(all_pred.reshape(-1), all_label.reshape(-1), average="weighted")
            if f1 > max_f1:
                max_f1 = f1
                torch.save(self.model.state_dict(), f"{self.config.output_pth}")
            self.logger.info(f"train_loss:{train_loss/train_num}")
            self.logger.info(f'accuracy: {acc:10.6f}')
            self.logger.info(f'f1 score: {f1:10.6f}')
            self.logger.info(f'recall_score: {recall:10.6f}')

    def test(self):
        test_loader = self.get_dataloader(self.config.X_test, self.config.Y_test, self.config)
        all_pred = []
        all_label = []
        self.model.eval()
        for batch_test_input, batch_test_label in test_loader:
            batch_test_input = batch_test_input.to(self.device)
            batch_test_label = batch_test_label.to(self.device)
            pred = self.model(batch_test_input, batch_test_label)
            pred = torch.argmax(pred, dim=-1)
            all_pred.append(pred)
            all_label.append(batch_test_label)

        all_pred = torch.tensor([item.cpu().detach().numpy() for item in all_pred])
        all_label = torch.tensor([item.cpu().detach().numpy() for item in all_label])
        f1 = f1_score(all_pred.reshape(-1), all_label.reshape(-1), average="weighted")
        acc = accuracy_score(all_pred.reshape(-1), all_label.reshape(-1))
        recall = recall_score(all_pred.reshape(-1), all_label.reshape(-1), average="weighted")
        self.logger.info(f'test_accuracy: {acc:10.6f}')
        self.logger.info(f'test_f1 score: {f1:10.6f}')
        self.logger.info(f'test_recall_score: {recall:10.6f}')


