# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
from detectron.src.utils import setup, EarlyStopping

logger = logging.getLogger("detectron2")


class Exp_Main:
    def __init__(self, cfg) -> None:
        # self.args = args
        self.cfg = cfg
        self.model = build_model(cfg)
        self.optimizer = build_optimizer(self.cfg, self.model)
        self.scheduler = build_lr_scheduler(self.cfg, self.optimizer)
        self.checkpointer = DetectionCheckpointer(
            self.model,
            self.cfg.OUTPUT_DIR,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.periodic_checkpointer = PeriodicCheckpointer(
            self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=self.max_iter
        )
        self.writers = (
            [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]
            if comm.is_main_process()
            else []
        )

    def get_evaluator(self, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
            return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def do_test(self, cfg, model):
        results = OrderedDict()
        for dataset_name in cfg.DATASETS.TEST:
            data_loader = build_detection_test_loader(cfg, dataset_name)
            evaluator = self.get_evaluator(
                cfg,
                dataset_name,
                os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name),
            )
            results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                logger.info(f"Evaluation results for {dataset_name} in csv format:")
                print_csv_format(results_i)
        if len(results) == 1:
            results = list(results.values())[0]
        return results

    def do_train(self, resume=False):
        self.model.train()
        early_stopping = EarlyStopping(patience=self.cfg.patience, verbose=True)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        start_iter = (
            self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume).get(
                "iteration", -1
            )
            + 1
        )
        data_loader = build_detection_train_loader(self.cfg)
        logger.info(f"Starting training from iteration {start_iter}")
        with EventStorage(start_iter) as storage:
            for data, iteration in zip(data_loader, range(start_iter, self.max_iter)):
                iteration = iteration + 1
                storage.step()

                # loss_dict = self.model(data)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        loss_dict = self.model(data)
                else:
                    loss_dict = self.model(data)

                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {
                    k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
                }
                losses_reduced = sum(loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                self.optimizer.zero_grad()
                if self.args.use_amp:
                    scaler.scale(losses).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    losses.backward()
                    self.optimizer.step()

                storage.put_scalar(
                    "lr", self.optimizer.param_groups[0]["lr"], smoothing_hint=False
                )
                self.scheduler.step()

                if (
                    self.cfg.TEST.EVAL_PERIOD > 0
                    and iteration % self.cfg.TEST.EVAL_PERIOD == 0
                    and iteration != self.max_iter
                ):
                    val_losses = self.do_test(self.cfg, self.model)
                    val_loss = sum(val_losses.values())
                    early_stopping(val_loss, self.model, "output")
                    comm.synchronize()
                    if early_stopping.early_stop:
                        logger.info("Early stopping")
                        break

                if iteration - start_iter > 5 and (
                    iteration % self.cfg.writer_period == 0
                    or iteration == self.max_iter
                ):
                    for writer in self.writers:
                        writer.write()
                self.periodic_checkpointer.step(iteration)


# early stopping,
# every 2k iterations evaluate


def main(args):
    cfg = setup(args)
    exp = Exp_Main(cfg)
    logger.info(f"Model:\n{exp.model}")

    if args.is_training:
        distributed = comm.get_world_size() > 1
        if distributed:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
        exp.do_train(resume=args.resume)

    else:
        DetectionCheckpointer(exp.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return exp.do_test(cfg, model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
