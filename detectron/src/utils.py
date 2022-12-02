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
