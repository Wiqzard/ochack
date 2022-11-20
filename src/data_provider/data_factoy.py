import torch
import os

from typing import Tuple, List, Dict
import json
from PIL import Image
from torchvision.io.image import read_image
import sys

from data_provider.data_set import ImageDataset

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
        self.transforms = transforms
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
        return image


class SordiAiDataset(ImageDataset):
    def __init__(
        self,
        root_path: str,
        data_path: str = "SORDI_2022_Single_Assets",
        transforms=None,
        flag="train",
    ) -> None:
        super().__init__(root_path, data_path, flag)
        self.transforms = transforms
        self.flag = flag

        self._set_directory()
        self.samples = self._make_dataset()

    def _make_dataset(self) -> List[Tuple[str, str]]:
        instances = []
        for directory in os.listdir(self.DIRECTORY):
            if directory.startswith(".") or directory.endswith("json"):
                continue
            directory = os.fsdecode(directory)
            # images, labels
            images_path = os.path.join(self.DIRECTORY, directory, "images")
            labels_path = os.path.join(self.DIRECTORY, directory, "labels/json")

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
            return img.convert("RGB")

    def json_loader(self, path: str) -> Dict:
        with open(path, "rb") as json_file:
            return json.load(json_file)

    def _transform_label(self):
        """
        transforms box dimensions if necessary
        transforms objects_ids
        """
        pass

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        # image after transform 3 ,720, 1280
        path_image, path_label = self.samples[index]
        image = self.image_loader(path_image)
        label = self.json_loader(path_label)
        image = self.transforms(image) if self.transforms else image

        return image, label[0]
