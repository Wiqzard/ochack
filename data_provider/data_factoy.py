import torch
import os

from typing import Tuple, List, Dict
import json
from PIL import Image
from torchvision.io.image import read_image
import sys
import torchvision.transforms as T
from data_provider.data_set import ImageDataset
from utils.tools import falsy_path
from utils.constants import CLASSES

sys.path.insert(0, "/Users/sebastian/Documents/Projects/sordi_ai/src")


class SordiAiDatasetEval(ImageDataset):
    def __init__(
        self,
        root_path: str,
        data_path: str = "eval",
        transforms=None,
        flag="eval",
    ) -> None:
        super().__init__(root_path, data_path, flag)
        self.height, self.width = 600, 1024
        self.transforms = transforms  # or T.Resize((self.height, self.width))
        self.flag = flag

        self._set_directory()
        self.samples = self._make_dataset()

    def _make_dataset(self) -> List[Tuple[str, str]]:
        instances = []
        images_path = os.path.join(self.DIRECTORY, "images")
        images_paths = sorted(os.listdir(images_path))
        for image in images_paths:
            if image.startswith("."):
                continue
            image = os.path.join(images_path, image)
            instances.append(image)
        return instances

    def get_raw_image(self, index: int) -> torch.Tensor:
        path_image, _ = self.samples[index]
        return read_image(path_image)

    def image_loader(self, path: str) -> Image.Image:
        _, image_name = os.path.split(path)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB"), image_name

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        path_image = self.samples[index]
        image, image_name = self.image_loader(path_image)
        image = self.transforms(image) if self.transforms else image
        image_width, image_height = 0, 0

        return image_name, image_width, image_height, image


class SordiAiDataset(ImageDataset):
    def __init__(
        self,
        root_path: str,
        data_path: str = "train",
        transforms=None,
        flag="train",
        ignore_redundant=False,
        partion_single_assets=False,
    ) -> None:
        super().__init__(root_path, data_path, flag)
        self.height, self.width = 600, 1024
        self.transforms = transforms  # or T.Resize((self.height, self.width))
        self.flag = flag
        self.ignore_redundant = ignore_redundant
        self.partion_single_assets = partion_single_assets
        self.redundant_directories = [
            "SORDI_2022_h4020_warehouse",
            "SORDI_2022_h4019_KLT stack",
            "SORDI_2022_h4018_KLT on rack",
        ]
        self.box_area_threshold = 1000
        self._set_directory()
        self.samples = self._make_dataset()

    def _make_dataset(self) -> List[Tuple[str, str]]:
        instances = []
        for dir in os.listdir(self.DIRECTORY):
            if falsy_path(directory=dir):
                continue
            path_to_data_part = os.path.join(self.DIRECTORY, dir)

            for directory in sorted(os.listdir(path_to_data_part)):
                if falsy_path(directory=directory):
                    continue
                if self.ignore_redundant and directory in self.redundant_directories:
                    continue
                directory = os.fsdecode(directory)
                # images, labels
                images_path = os.path.join(path_to_data_part, directory, "images")
                labels_path = os.path.join(path_to_data_part, directory, "labels/json")

                images_paths = sorted(os.listdir(images_path))
                labels_paths = sorted(os.listdir(labels_path))

                for image, label in zip(images_paths, labels_paths):
                    if image.startswith(".") or label.startswith("."):
                        continue
                    image_name, label_name = image.split(".")[0], label.split(".")[0]
                    if (
                        image_name == label_name
                        and image_name % self.partion_single_assets == 0
                    ):
                        image = os.path.join(images_path, image)
                        label = os.path.join(labels_path, label)
                        instances.append((image, label))

        return instances

    def get_raw_image(self, index: int) -> torch.Tensor:
        path_image, _ = self.samples[index]
        return read_image(path_image)

    def image_loader(self, path: str) -> Image.Image:

        _, file_name = os.path.split(path)
        file_name_raw = file_name.split(".")[0]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
            # img = self.transforms(img)
            return img, file_name_raw

    def json_loader(self, path: str) -> Dict:
        _, file_name = os.path.split(path)
        file_name_raw = file_name.split(".")[0]
        with open(path, "rb") as json_file:
            return json.load(json_file), file_name_raw

    def _transform_target(self, targets):
        """
        transforms box dimensions if necessary
        transforms objects_ids
        """
        scale_width = self.width / 1280
        scale_height = self.height / 720

        num_objs = len(targets)
        boxes = []
        labels = []
        for target in targets:
            x1, y1, x2, y2 = (
                target["Left"],
                target["Top"],
                target["Right"],
                target["Bottom"],
            )
            # if self.transforms:
            #     x1 = int(x1 * scale_width)
            #    y1 = int(y1 * scale_height)
            #    x2 = int(x2 * scale_width)
            #    y2 = int(y2 * scale_height)
            if x1 < x2 and y1 < y2 and (x2 - x1) * (y2 - y1) < self.box_area_threshold:

                boxes.append([x1, y1, x2, y2])
                labels.append(CLASSES[str(target["ObjectClassName"])])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return {"boxes": boxes, "labels": labels}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        # image after transform 3 ,720, 1280
        # 3, 600, 1024
        path_image, path_target = self.samples[index]
        image, image_name = self.image_loader(path_image)
        image = self.transforms(image) if self.transforms else image
        target, target_name = self.json_loader(path_target)
        target = self._transform_target(target)
        target["image_id"] = torch.tensor([index])
        ##ASSERT NAMES EQUAL i.e. 22 == 22
        assert image_name == target_name, "file missmatch"
        return image, target


# dataloader:
# [B, C, H, W],
# [B * Dict["labels": [NUM_OBJECTS], "boxes":[NUM_OBJECTS, 4]]]
