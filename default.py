
from yacs.config import CfgNode as CN
import argparse

_C=CN()
_C.MODEL = CN()
_C.MODEL.PRETRAIN_PATH = ""
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "resnet18"

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.SHUFFLE = True
_C.DATALOADER.TRAIN_BATCH_SIZE = 32
_C.DATALOADER.TEST_BATCH_SIZE = 32

_C.DATASET = CN()
_C.DATASET.ROOT = "./data"
_C.DATASET.TRAIN_LABEL_FILE = ""
_C.DATASET.TEST_LABEL_FILE = ""
_C.DATASET.INPUT_SIZE = 224

_C.SOLVER = CN()
_C.SOLVER.LR = 0.001
_C.SOLVER.LR_SCHEDULER_NAME = ""
_C.SOLVER.GAMMA = 0.5
_C.SOLVER.STEP_SIZE = 80

_C.TASK = CN()
_C.TASK.ROOT = "experiment"
_C.TASK.NAME = ""
_C.TASK.COORD_NUM = 8
_C.TASK.EPOCH = 200
_C.TASK.WEIGHT_DIR = "weight"
_C.TASK.LOG_DIR = "log"
_C.TASK.WRITER_FREQ = 1

_C.DEVICE = "cuda:0"




def get_cfg_defaults():
    return _C.clone() #局部变量使用形式