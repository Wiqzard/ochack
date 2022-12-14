import itertools
import json
import os
from typing import List, Dict
import shutil
from tqdm import tqdm
import argparse

# if ran from on directory up


CLASSES_ID = {
    1002: 1,
    1003: 2,
    1012: 3,
    1013: 4,
    1011: 5,
    1100: 6,
    1120: 7,
    2010: 8,
    2050: 9,
    2000: 10,
    1110: 11,
    4000: 12,
    5010: 13,
    1135: 14,
    1030: 15,
    1040: 16,
    1070: 17,
}
CLASSES_ID_RE = {value: key for key, value in CLASSES_ID.items()}
CLASSES_RE = {
    1: "stillage_close",
    2: "stillage_open",
    3: "l_klt_6147",
    4: "l_klt_8210",
    5: "l_klt_4147",
    6: "pallet",
    7: "jack",
    8: "forklift",
    9: "str",
    10: "bicycle",
    11: "dolly",
    12: "exit_sign",
    13: "fire_extinguisher",
    14: "spring_post",
    15: "locker",
    16: "cabinet",
    17: "cardboard_box",
}
models = {1: [1, 2, 3, 4, 5], 2: [6, 15, 16, 17, 9, 8], 3: [7, 10, 11, 12, 13, 14]}
model_classes = {
    1: {
        0: "stillage_close",
        1: "stillage_open",
        2: "l_klt_6147",
        3: "l_klt_8210",
        4: "l_klt_4147",
    },
    2: {
        0: "pallet",
        1: "locker",
        2: "cabinet",
        3: "cardboard_box",
        4: "str",
        5: "forklift",
    },
    3: {
        0: "jack",
        1: "bicycle",
        2: "dolly",
        3: "exit_sign",
        4: "fire_extinguisher",
        5: "spring_post",
    },
}
reversed_model_classes = {
    model_num: {value: key for key, value in classes.items()}
    for model_num, classes in model_classes.items()
}


def falsy_path(directory: str) -> bool:
    return bool(
        (
            directory.startswith(".")
            or directory.endswith("json")
            or directory.endswith("zip")
        )
    )


# def check_area2(box, area_threshold):
#    x1, y1, x2, y2 = (
#        box[0],
#        box[1],
#        box[2],
#        box[3],
#    )
#    return (x2 - x1) * (y2 - y1) > area_threshold


def make_dataset(
    root_path,
    target_path,
    partition_assets,
    area_min,
    area_max,
    mode: str = "train",
    model: int = 1,
) -> List[Dict]:
    """Create Folder dataset_{model}
    Create Folder train, eval
    In both create folder images, labels
    Choose image, based on labels
    Choose labels based on which classifications
    Transform boxes to xyhw
    Put image in train, label as txt in label"""

    # root_path/data/train or eval
    source_path = os.path.join(root_path, mode)

    # target_path/dataset_1,2,3/train, eval
    assert mode in {"train", "eval"}, "specified mode does not exist"
    model_path = f"dataset_{model}"
    destination_path = os.path.join(target_path, model_path)
    destination_path = os.path.join(destination_path, mode)
    os.makedirs(destination_path, exist_ok=True)
    # target_path/dataset_1/train/images, labels
    images_destination_path = os.path.join(destination_path, "images")
    labels_destination_path = os.path.join(destination_path, "labels")
    os.makedirs(images_destination_path, exist_ok=True)
    os.makedirs(labels_destination_path, exist_ok=True)

    if mode == "eval":
        make_eval_dataset(
            source_path, model, images_destination_path, labels_destination_path
        )

    else:
        make_train_dataset(
            source_path,
            partition_assets,
            area_min,
            area_max,
            model,
            images_destination_path,
            labels_destination_path,
        )


def get_annotations(
    label_path: str, model: int, area_min: int, area_max: int
) -> List[Dict]:
    image_width, image_height = 1280, 720
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
            if area_min > (x2 - x1) * (y2 - y1) or (x2 - x1) * (y2 - y1) > area_max:
                continue
            # prediction_class = CLASSES_ID[int(label["ObjectClassId"])]
            #            classes = model_classes[model]
            classes = reversed_model_classes[model]
            if label["ObjectClassName"] not in classes.keys():
                continue
            prediction_class = classes[label["ObjectClassName"]]
            # if prediction_class not in models[model]:
            #    continue
            w = (x2 - x1) / image_width
            h = (y2 - y1) / image_height
            x_center = (x1 + (x2 - x1) / 2) / image_width
            y_center = (y1 + (y2 - y1) / 2) / image_height
            # class x_center y_center width height
            annotation = {
                "prediction_class": prediction_class,
                "x_center": x_center,
                "y_center": y_center,
                "width": w,
                "height": h,
            }
            annotations.append(annotation)
    return annotations


def make_train_dataset(
    source_path,
    partition_assets,
    area_min,
    area_max,
    model,
    images_destination_path,
    labels_destination_path,
):
    idx = 1
    step = 1
    for dir in os.listdir(source_path):
        # dir Single_Assets Plant etc..
        if falsy_path(directory=dir):
            continue
        path_to_data_part = os.path.join(source_path, dir)

        for directory in sorted(os.listdir(path_to_data_part)):
            # Bicycle etc.
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
                if (
                    dir in ["SORDI_2022_Single_Assets", "SORDI_2022_Regensburg_plant"]
                    and step % partition_assets != 0
                ):
                    step += 1
                    continue
                image_name, label_name = image.split(".")[0], label.split(".")[0]

                if image_name == label_name:
                    #                    label = os.path.join((idx))
                    image_path = os.path.join(images_path, image)
                    label_path = os.path.join(labels_path, label)

                    annotations = get_annotations(label_path, model, area_min, area_max)
                    if len(annotations) == 0:
                        continue
                    image_destination_path = os.path.join(
                        images_destination_path, f"{str(idx)}.jpg"
                    )
                    label_destination_path = os.path.join(
                        labels_destination_path, f"{str(idx)}.txt"
                    )
                    if not os.path.exists(image_destination_path):
                        shutil.copy(image_path, image_destination_path)
                    write_labels(annotations, label_destination_path)
                    idx += 1
                    step += 1


def make_eval_dataset(
    source_path, model, images_destination_path, labels_destination_path
):
    idx = 1
    images_path = os.path.join(source_path, "images")
    labels_path = os.path.join(source_path, "labels")

    images_paths = sorted(os.listdir(images_path))
    labels_paths = sorted(os.listdir(labels_path))

    for image, label in zip(images_paths, labels_paths):
        if image.startswith(".") or label.startswith("."):
            continue
        image_name, label_name = image.split(".")[0], label.split(".")[0]

        assert image_name == label_name, "image, label missmatch"
        image_path = os.path.join(images_path, image)
        label_path = os.path.join(labels_path, label)

        image_destination_path = os.path.join(
            images_destination_path, f"{str(idx)}.jpg"
        )
        label_destination_path = os.path.join(
            labels_destination_path, f"{str(idx)}.txt"
        )
        if not delete_rows(label_path, label_destination_path, model):
            continue

        if not os.path.exists(os.path.join(image_destination_path)):
            shutil.copy(image_path, image_destination_path)
        idx += 1


def delete_rows(file_name, destination_name, model):

    if not os.path.exists(file_name):
        print("Error: file does not exist.")
        return False

    with open(file_name, "r") as input_file:
        lines = input_file.readlines()

    with open(destination_name, "w+") as output_file:
        num_lines = 0
        for line in lines:
            values = line.split(" ")
            if int(values[0]) in models[model]:
                mapped_class = str(
                    reversed_model_classes[model][CLASSES_RE[int(values[0])]]
                )
                new_line = f"{mapped_class} " + " ".join(values[1:])
                output_file.write(new_line)
                num_lines += 1
        if num_lines != 0:
            return True
        if os.path.exists(destination_name):
            os.remove(destination_name)
        return False


def write_labels(annotations, label_destination_path):
    for annotation in annotations:
        with open(label_destination_path, "a+") as text_writer:
            dict_entry = {
                "prediction_class": annotation["prediction_class"],
                "x_center": annotation["x_center"],
                "y_center": annotation["y_center"],
                "w": annotation["width"],
                "h": annotation["height"],
            }
            text = " ".join(map(str, list(dict_entry.values()))) + "\n"

            text_writer.write(text)


def make_all_datasets(args):
    modes = ["train", "eval"]
    with tqdm(total=6, leave=True) as pbar:
        for model, mode in itertools.product(models.keys(), modes):
            make_dataset(
                args.source,
                args.destination,
                partition_assets=args.partition_assets,
                area_min=args.area_threshold_min,
                area_max=args.area_threshold_max,
                mode=mode,
                model=model,
            )
            pbar.update(1)


# make_dataset("data/test_source", "data/test_destination", mode="train", model=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        default="data/test_source",
        help="source data path ",
    )
    parser.add_argument(
        "--destination",
        type=str,
        required=True,
        default="data/test_destinatin",
        help="destination data path",
    )
    parser.add_argument(
        "--area_threshold_min",
        type=int,
        required=False,
        default=2000,
        help="The minimum area threshold",
    )
    parser.add_argument(
        "--area_threshold_max",
        type=int,
        required=False,
        default=800000,
        help="The maximum area threshold",
    )
    parser.add_argument(
        "--partition_assets",
        type=int,
        required=False,
        default=2,
        help="The number of partitioned assets",
    )

    args = parser.parse_args()
    make_all_datasets(args)


if __name__ == "__main__":
    main()
