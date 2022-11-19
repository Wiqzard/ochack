import torch
import torchvision
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from data_factory.data_factoy import SordiAiDataset
from config import config
from network.pretrained import create_model, weights  # , model
from typing import Dict, List


import argparse
import os
import torch

from exp.exp_main import ExpIc

parser = argparse.ArgumentParser(description="Warehouse [Object Classification]")
parser.add_argument()

args = parser.parse_args()

args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)

print("Args in experiment:")
print(args)


# classes = {
#    1002: 0,
#    1003: 1,
#    1012: 2,
#    1013: 3,
#    1011: 4,
#    1100: 5,
#    1120: 6,
#    2010: 7,
#    2050: 8,
#    2000: 9,
#    1110: 10,
#    4000: 11,
#    5010: 12,
#    1135: 13,
#    1030: 14,
#    1040: 15,
#    1070: 16,
# }
classes = {
    "__background__": 0,
    "stillage_close": 1,
    "stillage_open": 2,
    "l_klt_6147": 3,
    "l_klt_8210": 4,
    "l_klt_4147": 5,
    "pallet": 6,
    "jack": 7,
    "forklift": 8,
    "str": 9,
    "bicycle": 10,
    "dolly": 11,
    "exit_sign": 12,
    "fire_extinguisher": 13,
    "spring_post ": 14,
    "locker": 15,
    "cabinet": 16,
    "cardboard_box": 17,
}
model = create_model(len(classes))

preprocess = weights.transforms()
train_dataset = SordiAiDataset(transforms=preprocess)
print(preprocess)
train_dataloder = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=config["shuffle"],
    num_workers=config["num_workers"],
    drop_last=config["drop_last"],
)
train_dataset[0]


def transform_label(
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
    print(targets)
    return targets  # labels


def train_model() -> None:
    model.train()
    """
    input:
    images, targets   -> 
    [Batch, C, H, W],   
    [Batch * Dict["boxes": [x1,y1,x2,y2], "labels": [label1, label2,...]]
    """
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"{total_trainable_params:,} training parameters.\n")
    for image, label in train_dataloder:
        target = transform_label(label)  # torch.tensor([1])
        prediction = model(image, target)
        loss_classifier = prediction["loss_classifier"]
        loss_box_re = prediction["loss_box_re"]
        loss_objectnes = prediction["loss_objectnes"]
        loss_rpn_box_reg = prediction["loss_rpn_box_reg"]
        break
    #    image, label = train_dataset[0]
    #    image = image  # .unsqueeze(0)
    #    boxes = torch.tensor(
    #        [label["Left"], label["Top"], label["Right"], label["Bottom"]]
    #    )  # .unsqueeze(0)
    #    labels = torch.tensor([classes[label["ObjectClassId"]]])  # torch.tensor([1])
    #    targets = [{"boxes": boxes, "labels": labels}]
    #    prediction = model(image, targets)
    print(prediction)


train_model()


def show_prediction(index: int) -> None:
    model.eval()
    image, _ = train_dataset[index]
    img = train_dataset.get_raw_image(index)
    prediction = model(image.unsqueeze(0))[0]
    print(prediction)
    labels = [weights.meta["categories"][i] for i in prediction["labels"]]
    box = draw_bounding_boxes(
        img,
        boxes=prediction["boxes"],
        labels=labels,
        colors="red",
        width=4,
        font_size=30,
    )
    im = to_pil_image(box.detach())
    im.show()


# for i in range(len(train_dataset)):
#    print(train_dataset[i][1])
# for image, label in train_dataloder:
#    print(label)
#    break
# batch = [preprocess(image)]
# prediction = model(batch)[0]
# labels = [weights.meta["categories"][i] for i in prediction["labels"]]
# box = draw_bounding_boxes(
#     img,
#     boxes=prediction["boxes"],
#     labels=labels,
#     colors="red",
#     width=4,
#     font_size=30,
# )
# im = to_pil_image(box.detach())
# im.show()
# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
# predictions = model(x)
# print(predictions)
#
