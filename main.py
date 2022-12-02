import random
import torch
import argparse
import numpy as np
import os
import logging
import warnings

from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog


from detectron.src.utils import setup
from detectron.src.trainer import MyTrainer
from detectron.src.data_set import DataSet, register_dataset
from detectron.src.constants import CLASSES

warnings.filterwarnings("ignore")
logger = logging.getLogger("__name__")
level = logging.INFO
logger.setLevel(level)
ch = logging.StreamHandler()
ch.setLevel(level)
logger.addHandler(ch)


def main():  # sourcery skip: extract-method
    fix_seed = 1401
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="Warehouse [Object Classification]")

    parser.add_argument(
        "--is_training", type=bool, required=True, default=True, help="status"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume training from checkpoint",
    )
    parser.add_argument(
        "--writer_period", type=int, required=False, default=100, help="period to log"
    )
    #   <------------- data loader ------------->
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/",
        help="root path of the data directory",
    )
    parser.add_argument(
        "--ignore_redundant",
        action="store_true",
        default=False,
        help="ignore warehouse, on stack, on rack data",
    )
    parser.add_argument(
        "--partion_single_assets",
        action="store_true",
        default=False,
        help="only use every n'th image from single assets",
    )
    parser.add_argument(
        "--ratio", type=float, default=0.8, help="train-test split ratio"
    )
    parser.add_argument(
        "--area_threshold",
        type=int,
        default=3000,
        help="sort out boxes with smaller area",
    )

    #   <------------- trainer ------------->
    # parser.add_argument(
    #     "--datasets_train", type=str, default="data_train", help="train dataset name"
    # )
    # parser.add_argument(
    #     "--datasets_test", type=str, default="data_val", help="validation dataset name"
    # )
    parser.add_argument(
        "--model",
        type=str,
        default="COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml",
        help="modelzoo zaml file",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="dataloader number of workers"
    )
    parser.add_argument("--ims_per_batch", type=int, default=2, help="batch size")
    parser.add_argument("--base_lr", type=float, default=0.003, help="learning rate")
    parser.add_argument(
        "--max_iter", type=int, default=3000, help="iterations per epoch"
    )
    parser.add_argument(
        "--batch_per_img", type=int, default=512, help="roi heads per image"
    )
    # parser.add_argument(
    #    "--num_classes", type=int, default=17, help="number of classes"
    # )
    parser.add_argument("--patience", type=int, default=1000, help="early stopping")
    parser.add_argument(
        "--eval_period", type=int, default=100, help="after periods evaluate model"
    )
    parser.add_argument(
        "--checkpoint_period", type=int, default=100, help="checkpoint after n periods"
    )
    parser.add_argument("--use_gpu", action="store_true", default=True, help="use gpu")

    args = parser.parse_args()
    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)

    logger.info("Args in experiment:")
    logger.info(args)

    cfg = setup(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if args.is_training:
        logger.info(f">>>>>>> start training : {args.model} >>>>>>>>>>>>>>>>>>>>>>>>>>")
        dataset = DataSet(args)
        for d in ["train", "val"]:
            logger.info(f">>>>>>> registering data_{d} >>>>>>> ")
            DatasetCatalog.register(
                f"data_{d}", lambda d=d: dataset.dataset_function(mode=d)
            )
            MetadataCatalog.get(f"data_{d}").set(thing_classes=CLASSES)

        trainer = MyTrainer(cfg)
        trainer.resume_or_load(resume=args.resume)
        trainer.train()


if __name__ == "__main__":
    main()
