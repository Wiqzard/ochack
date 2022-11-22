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
        self.transforms = transforms or T.Resize((self.height, self.width))
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
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        # image after transform 3 ,720, 1280
        path_image = self.samples[index]
        image = self.image_loader(path_image)
        image = self.transforms(image) if self.transforms else image
        # print(image.shape)
        image_name, _ = os.path.split(path_image)
        image_width, image_height = 0, 0

        return image_name, image_width, image_height, image


class SordiAiDataset(ImageDataset):
    def __init__(
        self,
        root_path: str,
        data_path: str = "train",
        transforms=None,
        flag="train",
    ) -> None:
        super().__init__(root_path, data_path, flag)
        self.height, self.width = 600, 1024
        self.transforms = transforms or T.Resize((self.height, self.width))
        self.flag = flag

        self._set_directory()
        self.samples = self._make_dataset()

    def _make_dataset(self) -> List[Tuple[str, str]]:
        instances = []
        for dir in os.listdir(self.DIRECTORY):
            if falsy_path(directory=dir):
                continue
            path_to_data_part = os.path.join(self.DIRECTORY, dir)

            for directory in os.listdir(path_to_data_part):
                if falsy_path(directory=directory):
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
                    image = os.path.join(images_path, image)
                    label = os.path.join(labels_path, label)
                    instances.append((image, label))
        return instances

    def get_raw_image(self, index: int) -> torch.Tensor:
        path_image, _ = self.samples[index]
        return read_image(path_image)

    def image_loader(self, path: str) -> Image.Image:
        with open(path, "rb") as f:
            img = Image.open(f)
            img = self.transforms(img)
            return img.convert("RGB")

    def json_loader(self, path: str) -> Dict:
        with open(path, "rb") as json_file:
            return json.load(json_file)

    def _transform_label(self, label):
        """
        transforms box dimensions if necessary
        transforms objects_ids
        """
        # sourcery skip: inline-immediately-returned-variable
        x1, y1, x2, y2 = label["Left"], label["Top"], label["Right"], label["Bottom"]
        scale_width = self.width / 1280
        scale_height = self.height / 720
        x1 *= scale_width
        y1 *= scale_height
        x2 *= scale_width
        y2 *= scale_height
        targets = [
            {
                "boxes": torch.tensor(
                    [label["Left"], label["Top"], label["Right"], label["Bottom"]]
                ).unsqueeze(0),
                "labels": torch.tensor([CLASSES[str(label["ObjectClassName"])]]),
            }
        ]
        return targets

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        # image after transform 3 ,720, 1280
        # 3, 600, 1024
        path_image, path_label = self.samples[index]
        image = self.image_loader(path_image)
        label = self.json_loader(path_label)[0]
        # image = self.transforms(image)
        # print(self.transforms)
        # print(type(image))
        label = self._transform_label(label)
        return image, label
