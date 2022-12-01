from detectron2 import model_zoo
from detectron2.config import get_cfg


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
    cfg.DATASETS.TRAIN = "data_train"
    cfg.DATASETS.TEST = "data_val"
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        args.model
    )  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.max_iter
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (
        args.batch_per_image
    )  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 17
    cfg.TEST.EVAL_PERIOD = args.eval_period
    return cfg
