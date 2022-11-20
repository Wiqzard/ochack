import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from data_provider.data_factoy import SordiAiDataset, SordiAiDatasetEval
from utils.config import config
from network.pretrained import create_model, weights  # , model
from typing import Dict, List

from utils.tools import transform_label, train_test_split
from utils.constants import CLASSES
from exp.exp_main import Exp_Main


model, weights = create_model(len(CLASSES))

preprocess = weights.transforms()

full_dataset = SordiAiDataset(root_path="./data/", transforms=preprocess)
train_dataset, test_dataset = train_test_split(full_dataset, "train"), train_test_split(
    full_dataset, "test"
)
eval_dataset = SordiAiDatasetEval(root_path="./data/", transforms=preprocess)
train_dataloder = DataLoader(
    train_dataset,
    batch_size=3,
    shuffle=config["shuffle"],
    num_workers=config["num_workers"],
    drop_last=config["drop_last"],
)
print(train_dataset[0])
print(test_dataset[0])
print(eval_dataset[0])


def transform_label(
    labels: Dict,
) -> List[Dict]:  # sourcery skip: inline-immediately-returned-variable
    targets = [
        {
            "boxes": torch.tensor([x1, y1, x2, y2]).unsqueeze(0),
            "labels": torch.tensor([CLASSES[str(label)]]),
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
        # loss_classifier = prediction["loss_classifier"]
        # loss_box_re = prediction["loss_box_re"]
        # loss_objectnes = prediction["loss_objectnes"]
        # loss_rpn_box_reg = prediction["loss_rpn_box_reg"]

        #    image, label = train_dataset[0]
        #    image = image  # .unsqueeze(0)
        #    boxes = torch.tensor(
        #        [label["Left"], label["Top"], label["Right"], label["Bottom"]]
        #    )  # .unsqueeze(0)
        #    labels = torch.tensor([classes[label["ObjectClassId"]]])  # torch.tensor([1])
        #    targets = [{"boxes": boxes, "labels": labels}]
        #    prediction = model(image, targets)
        print(target)
        print(prediction)


# train_model()
