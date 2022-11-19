from data_provider.data_factoy import SordiAiDataset
from network.pretrained import create_model
from exp.exp_basic import Exp_Basic
from utils.tools import transform_label, adjust_learning_rate, EarlyStopping, logger
from utils.constants import CLASSES

# from utils.tools import EarlyStopping, adjust_learning_rate
# from utils.metrics import metric

import time
import numpy as np
import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader


class Exp_Main(Exp_Basic):
    classes = CLASSES

    def __init__(self, args) -> None:
        super().__init__(args)

    def _build_model(self):
        model_dict = {"faster_rcnn": 1}
        model, self.weights = create_model(len(self.classes))
        model = model.float().to(self.device)
        return model

    def _get_data(self, flag: str = "train") -> DataLoader:
        args = self.args

        if flag == "test":
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
        elif flag == "pred":
            shuffle_flag = False
            drop_last = False
            batch_size = 1
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size

        preprocess = self.weights.transforms()
        data_set = SordiAiDataset(
            root_path=args.root_path,
            data_path=args.data_path,
            transforms=preprocess,
            flag=flag,
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
        )
        return data_set, data_loader

    def _select_optimizer(self):
        # sourcery skip: inline-immediately-returned-variable
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # sourcery skip: inline-immediately-returned-variable
        criterion = nn.MSELoss()
        return criterion

    def validation(self, validation_data, validation_loader, criterion) -> float:
        self.model.train()  # train????
        total_loss = []
        for image, label in validation_loader:
            target = transform_label(classes=self.classes, labels=label)
            loss_dict = self.model(image, target)
            losses = sum(loss_dict.values())
            total_loss.append(losses.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        validation_data, validation_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (image, label) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                target = transform_label(classes=self.classes, labels=label)
                loss_dict = self.model(image, target)
                loss = sum(loss_dict.values())
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    logger.info(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    logger.info(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            logger.info(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.validation(validation_loader=validation_loader)

            logger.info(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = f"{path}/checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
