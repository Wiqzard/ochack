from typing import List, Dict, Tuple, Optional
import numpy as np
import logging
import time
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

from data_provider.data_factoy import SordiAiDataset

logger = logging.getLogger("__name__")
level = logging.INFO
logger.setLevel(level)
ch = logging.StreamHandler()
ch.setLevel(level)
logger.addHandler(ch)


def train_test_split(
    dataset: SordiAiDataset, ratio: float = 0.8, flag: str = "train"
) -> Tuple[SordiAiDataset, SordiAiDataset]:
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    if flag == "train":
        return train_dataset
    elif flag == "test":
        return test_dataset
    else:
        raise NameError


def transform_label(
    classes: Dict,
    labels: Dict,
) -> List[Dict]:  # sourcery skip: inline-immediately-returned-variable
    targets = [
        {
            "boxes": torch.tensor([x1, y1, x2, y2]).unsqueeze(0),
            "labels": torch.tensor([classes[str(label)]]),
        }
        for (x1, y1, x2, y2), label in zip(
            zip(labels["Left"], labels["Top"], labels["Right"], labels["Bottom"]),
            labels["ObjectClassName"],
        )
    ]
    # targets = {
    #     "boxes": torch.tensor([
    #         torch.tensor([x1, y1, x2, y2]).unsqueeze(0)
    #         for x1, y1, x2, y2 in zip(b["Left"], b["Top"], b["Right"], b["Bottom"])
    #     ]),
    #     "labels": torch.tensor([classes[int(label)] for label in b["ObjectClassId"]]),
    # }
    # print(targets["boxes"].shape)
    return targets


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f"Updating learning rate to {lr}")


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), f"{path}/checkpoint.pth")
        self.val_loss_min = val_loss


def log_train_progress(args, time_now, loss, epoch, train_steps, i, iter_count) -> None:
    logger.info(
        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item())
    )
    speed = (time.time() - time_now) / iter_count
    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
    logger.info("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))


def log_train_epoch(epoch, train_steps, train_loss, vali_loss) -> None:
    logger.info(
        "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss
        )
    )


# def show_prediction(image, index: int, dataset: Optional) -> None:
#    model.eval()
#    image, _ = train_dataset[index]
#    img = train_dataset.get_raw_image(index)
#    prediction = model(image.unsqueeze(0))[0]
#    print(prediction)
#    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
#    box = draw_bounding_boxes(
#        img,
#        boxes=prediction["boxes"],
#        labels=labels,
#        colors="red",
#        width=4,
#        font_size=30,
#    )
#    im = to_pil_image(box.detach())
#    im.show()
#
