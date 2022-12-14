




from utils import setup
from data_set import DataSet, register_dataset
from constants import CLASSES
from trainer import MyTrainer
from custom_train import main
from detectron2.engine import launch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
import torch
from src.utils import dotdict
from utils.tools import logger


import sys 
import warnings

if not 'detectron' in sys.path:
  print("added")
  sys.path += ['detectron']
  
  
args = dotdict()
args.is_training = True
args.writer_period = 10
args.patience = 10 
args.root_path = "/content/drive/MyDrive/Datasets/"
args.ignore_redundant = True
args.partion_single_assets = True


args.area_threshold_min = 5000 
args.area_threshold_max = 800000
args.overlap_threshold = 0 #0.1

args.model = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
args.num_workers = 4
args.ims_per_batch = 2
args.ratio = 0.9
args.base_lr = 0.000
args.max_iter = 1000 
args.batch_per_img = 4096 
args.eval_period = 50 
args.use_gpu = True
args.checkpoint_period = 10
args.use_amp = False
args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
logger.info("Args in experiment:")
logger.info(args)



dataset = DataSet(args)
for d in ["train", "val"]:
    logger.info(f">>>>>>> registering data_{d} >>>>>>> ")
    DatasetCatalog.register(
        f"data_{d}", lambda d=d: dataset.dataset_function(mode=d)
    )
    MetadataCatalog.get(f"data_{d}").set(thing_classes=CLASSES)
    
    
from detectron2.data import DatasetCatalog, MetadataCatalog
import random
import cv2
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
# get the registered dataset
dataset_dicts = DatasetCatalog.get("data_train")

# get the metadata for the dataset
metadata = MetadataCatalog.get("data_train")

# visualize a sample from the dataset
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imshow(vis.get_image()[:, :, ::-1])
    
    
    