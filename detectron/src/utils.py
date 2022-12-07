from detectron2 import model_zoo
from detectron2.config import get_cfg

import torch
import numpy as np


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def falsy_path(directory: str) -> bool:
    return bool(
        (
            directory.startswith(".")
            or directory.endswith("json")
            or directory.endswith("zip")
        )
    )


def check_bounding_box(args, bbox, annotations):
    x1, y1, x2, y2 = bbox

    if x1 == x2 or y1 == y2:
        return False
    if not (args.area_threshold_max > (x2 - x1) * (y2 - y1) > args.area_threshold_min):
        return False

    if len(annotations) == 0:
        return True

    bboxes = [annotation["bbox"] for annotation in annotations]
    for b in bboxes:
        ox1 = max(x1, b[0])
        oy1 = max(y1, b[1])
        ox2 = min(x2, b[2])
        oy2 = min(y2, b[3])
        if ox1 < ox2 and oy1 < oy2:
            overlap_area = (ox2 - ox1) * (oy2 - oy1)
            total_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            print(30 * "-")
            print(ox1)
            print(oy1)
            print(ox2)
            print(oy2)
            print(overlap_area)
            print(total_area)
            print(overlap_area / total_area)
            if overlap_area / total_area < args.overlap_threshold:
                return True
        else:
            return True
    return False


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.model))

    cfg.DATASETS.TRAIN = ("data_train",)
    cfg.DATASETS.TEST = ("data_val",)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        args.model
    )  # Let training initialize from model zoo
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_per_img
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17
    if not args.use_gpu:
        cfg.MODEL.DEVICE = "cpu"

    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period

    cfg.TEST.EVAL_PERIOD = args.eval_period

    cfg.writer_period = args.writer_period
    cfg.patience = args.patience
    cfg.use_amp = args.use_amp
    cfg.ratio = args.ratio
    cfg.ignore_redundant = args.ingore_reduntant
    cfg.partion_single_assets = args.partion_single_assets
    cfg.area_threshold_min = args.area_threshold_max
    cfg.area_threshold_max = args.area_threshold_max
    cfg.root_path = args.root_path

    return cfg


# def show_image():
#    dataset_dicts = dataset_function("train")
#    for d in random.sample(dataset_dicts, 3):
#        img = cv2.imread(d["file_name"])
#        visualizer = Visualizer(img[:, :, ::-1], metadata=train_metadata, scale=0.5)
#        out = visualizer.draw_dataset_dict(d)
#        cv2_imshow(out.get_image()[:, :, ::-1])


class EarlyStopping:
    def __init__(self, patience=1, verbose=False, delta=0):
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
