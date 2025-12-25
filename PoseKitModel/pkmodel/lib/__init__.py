import os

from lib.data.data import Data
from lib.models import PoseKitModel
from lib.task import Task
from lib.utils import setRandomSeed, printDash
import torch.multiprocessing as mp

def init(cfg):

    if cfg["cfg_verbose"]:
        printDash()
        print(cfg)
        printDash()

    try:
        mp.set_sharing_strategy("file_system")
    except RuntimeError:
        # Strategy already set or unsupported; keep default.
        pass

    gpu_id = str(cfg.get("GPU_ID", "")).strip()
    if gpu_id and gpu_id.lower() != "mps":
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    setRandomSeed(cfg['random_seed'])

    if not os.path.exists(cfg['save_dir']):
        os.makedirs(cfg['save_dir'])
