from typing import List, Dict, Tuple, Optional
import numpy as np
import logging
import time
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import pandas as pd
import cv2
from utils.constants import CLASSES, CLASSES_ID
import random

# from data_provider.data_factoy import SordiAiDataset

logger = logging.getLogger("__name__")
level = logging.INFO
logger.setLevel(level)
ch = logging.StreamHandler()
ch.setLevel(level)
logger.addHandler(ch)


def train_test_split(dataset, ratio: float = 0.8):
    train_size = int(ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    return train_dataset, test_dataset


def collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


# if flag == "train":
#     return train_dataset
# elif flag == "test":
#     return test_dataset
# else:
#     raise NameError
class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def transform_label(
    classes: Dict,
    labels: Dict,
) -> List[Dict]:  # sourcery skip: inline-immediately-returned-variable
    # targets = [
    #    {
    #        "boxes": torch.tensor([x1, y1, x2, y2]).unsqueeze(0),
    #        "labels": torch.tensor([classes[str(label)]]),
    #    }
    #    for (x1, y1, x2, y2), label in zip(
    #        zip(labels["Left"], labels["Top"], labels["Right"], labels["Bottom"]),
    #        labels["ObjectClassName"],
    #    )
    # ]

    return targets


def transform_target(targets) -> List[Dict]:
    """Dict of list to list of dicts"""
    return [dict(zip(targets, t)) for t in zip(*targets.values())]


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


def log_train_epoch(epoch, train_steps, train_loss, test_loss) -> None:
    logger.info(
        "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Test Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, test_loss
        )
    )


def log_loss(loss) -> None:
    logger.info(
        f"""Classifier Loss: {loss["loss_classifier"]} --- Box-Reg Loss: {loss["loss_box_reg"]}  \n
            Objectness Loss: {loss["loss_objectness"]} --- RPN-Box-Reg Loss: {loss["loss_rpn_box_reg"]} """
    )


def falsy_path(directory: str) -> bool:
    return bool(
        (
            directory.startswith(".")
            or directory.endswith("json")
            or directory.endswith("zip")
        )
    )


import csv


def write_to_csv(idx, image_name, image_width, image_height, label) -> None:
    """writes prection to csv,
    label: {'boxes': tensor([], size=(0, 4)),
             'labels': tensor([], dtype=torch.int64),
             'scores': tensor([]}
    """
    num_predictions = label["labels"].shape[0]
    with open("src/output/" + "submission.csv", "a") as submission:
        csv_writer = csv.writer(submission, delimiter=",")
        for i in range(num_predictions):
            label_num = label["labels"][i].item()
            if label_num != 0:
                idx += 1
                object_class_id = CLASSES_ID.index(label_num)
                object_class_name = CLASSES.index(label_num - 1)
                boxes = label["boxes"][i, :]
                score = label["scores"][i].item()

                row = {
                    "detection_id": idx,
                    "image_name": image_name,
                    "image_width": image_width,
                    "image_height": image_height,
                    "object_class_id": object_class_id,
                    "object_class_name": object_class_name,
                    "bbox_left": boxes[0].item(),
                    "bbox_top": boxes[1].item(),
                    "bbox_right": boxes[2].item(),
                    "bbox_bottom": boxes[3].item(),
                    "confidence": score,
                }
                csv_writer.writerow(row)
    return idx
    # print(type(img))


def show_tranformed_image(train_dataset, classes):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the tranformed images along with the corresponding
    labels are correct or not.

    """
    colors = np.random.uniform(0, 1, size=(len(CLASSES), 3))
    for i in range(2):
        index = random.randint(0, len(train_dataset) - 1)
        images, targets = train_dataset[index]  # next(iter(train_loader))
        #        targets = transform_target(targets)
        print(targets)
        boxes = targets[i]["boxes"].cpu().numpy().astype(np.int32)
        labels = targets[i]["labels"].cpu().numpy().astype(np.int32)
        print(boxes)
        print(labels)
        # Get all the predicited class names.
        classes_rev = {v: k for k, v in classes.items()}
        pred_classes = classes_rev[labels]  # [classes_rev[label] for label in labels]
        sample = images.permute(1, 2, 0).cpu().numpy()
        sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
        for box_num, box in enumerate(boxes):
            class_name = pred_classes[box_num]
            color = colors[box_num]
            cv2.rectangle(
                sample, (box[0], box[1]), (box[2], box[3]), color, 2, cv2.LINE_AA
            )
            cv2.putText(
                sample,
                class_name,
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.imshow("Transformed image", sample)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# row["detection_id"]
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
