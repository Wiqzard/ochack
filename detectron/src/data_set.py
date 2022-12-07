import json
from typing import List, Dict
import os

from detectron2.structures.boxes import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron.src.utils import falsy_path, check_bounding_box
from detectron.src.constants import CLASSES, CLASSES_DICT


class DataSet:
    def __init__(self, args):
        self.args = args
        self.redundant_directories = [
            "SORDI_2022_h4020_warehouse",
            "SORDI_2022_h4019_KLT stack",
            "SORDI_2022_h4018_KLT on rack",
        ]
        self.DIRECTORY = os.path.join(args.root_path, "train")  # "eval"

    def get_annotations(self, label_path: str) -> List[Dict]:
        with open(label_path, "rb") as json_file:
            labels = json.load(json_file)
            annotations = []
            for label in labels:
                x1, y1, x2, y2 = (
                    label["Left"] + 1,
                    label["Top"] + 1,
                    label["Right"] + 1,
                    label["Bottom"] + 1,
                )

                if (
                    True
                ):  # check_bounding_box(self.args, (x1, y1, x2, y2), annotations):
                    object_id = CLASSES_DICT[str(label["ObjectClassName"])]
                    annotation = {
                        "bbox": [x1, y1, x2, y2],
                        "bbox_mode": BoxMode.XYXY_ABS,
                        "category_id": object_id,
                        "iscrowd": 0,
                    }
                    annotations.append(annotation)
        return annotations

    def dataset_function(self, mode: str = "train") -> List[Dict]:
        args = self.args
        instances = []
        idx = 0
        for dir in os.listdir(self.DIRECTORY):
            if falsy_path(directory=dir):
                continue
            path_to_data_part = os.path.join(self.DIRECTORY, dir)
            for directory in sorted(os.listdir(path_to_data_part)):
                if falsy_path(directory=directory):
                    continue
                if args.ignore_redundant and directory in self.redundant_directories:
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

                    if (
                        dir == "SORDI_2022_Single_Assets"
                        and idx % args.partion_single_assets != 0
                    ):
                        idx += 1
                        continue
                    image_name, label_name = image.split(".")[0], label.split(".")[0]

                    if image_name == label_name:
                        image = os.path.join(images_path, image)
                        label = os.path.join(labels_path, label)

                        height, width = 720, 1280  # cv2.imread(image).shape[:2]

                        dict_entry = {
                            "file_name": image,
                            "image_id": idx,
                            "height": height,
                            "width": width,
                            "annotations": self.get_annotations(label),
                        }
                        n = int(1 / (1 - args.ratio))
                        if idx % n != 0 and mode == "train":
                            instances.append(dict_entry)
                        elif idx % n == 0 and mode == "val":
                            instances.append(dict_entry)
                        idx += 1
        return instances


def register_dataset(dataset):
    for d in ["train", "val"]:
        DatasetCatalog.register(f"data_{d}", lambda d=d: dataset_function(mode=d))
        MetadataCatalog.get(f"data_{d}").set(thing_classes=CLASSES)
    # train_metadata = MetadataCatalog.get("data_train")
