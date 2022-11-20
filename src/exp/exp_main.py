from data_provider.data_factoy import SordiAiDataset, SordiAiDatasetEval
from network.pretrained import create_model
from exp.exp_basic import Exp_Basic
from utils.tools import (
    transform_label,
    adjust_learning_rate,
    EarlyStopping,
    logger,
    log_train_progress,
    log_train_epoch,
    train_test_split,
)
from utils.constants import CLASSES

# from utils.tools import EarlyStopping, adjust_learning_rate
# from utils.metrics import metric
from typing import Tuple, Optional, List, Union
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

    def _get_data(
        self, flag: str = "train"
    ) -> Tuple[Union[SordiAiDataset, SordiAiDatasetEval], DataLoader]:
        args = self.args
        preprocess = self.weights.transforms()
        if flag == "eval":
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            data_set = SordiAiDatasetEval(
                root_path=args.root_path,
                data_path=args.data_path,
                transforms=preprocess,
                flag=flag,
            )
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size

            full_dataset = SordiAiDataset(
                root_path=args.root_path,
                data_path=args.data_path,
                transforms=preprocess,
                flag=flag,
            )
            train_dataset, test_dataset = train_test_split(
                full_dataset, args.ratio, flag
            )
            data_set = train_dataset if flag == "train" else test_dataset
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

    def test(self, test_data, test_loader, criterion) -> float:
        self.model.train()  # train????
        total_loss = []
        with torch.no_grad():
            for image, label in test_loader:
                loss = self._process_one_batch(image=image, label=label)
                total_loss.append(loss.item())
            total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def evaluation(self, validation_data, validation_loader, criterion) -> float:
        self.model.eval()
        labels = []
        for image in validation_loader:
            label = self.model(image)
            labels.append(label)
        return labels

    def _set_checkpoint(self, setting) -> str:
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def _process_one_batch(self, image, label):
        target = transform_label(classes=self.classes, labels=label)
        loss_dict = self.model(image, target)
        return sum(loss_dict.values())

    def train(self, setting):  # sourcery skip: low-code-quality
        train_data, train_loader = self._get_data(flag="train")
        test_data, test_loader = self._get_data(flag="test")

        path = self._set_checkpoint(setting=setting)
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
                loss = self._process_one_batch(image=image, label=label)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    log_train_progress(
                        args=self.args,
                        time_now=time_now,
                        loss=loss,
                        train_steps=train_steps,
                        i=i,
                        iter_count=iter_count,
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
            test_loss = self.test(test_loader=test_loader)
            log_train_epoch(epoch=epoch, train_steps=train_steps, vali_loss=test_loss)

            early_stopping(test_loss, self.model, path)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
            break
        best_model_path = f"{path}/checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model
