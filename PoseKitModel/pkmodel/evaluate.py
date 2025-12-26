import os
import argparse
import random
import torch

from lib import init, Data, PoseKitModel, Task
from config_loader import load_config

def _resolve_weights_path(cfg, cli_path):
    if cli_path:
        return cli_path
    cfg_path = str(cfg.get("eval_weights", "")).strip()
    if cfg_path:
        return cfg_path
    last_path = os.path.join(cfg.get("save_dir", "../output/"), "last.pth")
    return last_path

def _load_weights(run_task, path):
    if not path:
        return
    # Reuse checkpoint loader (handles both full checkpoint and pure state_dict)
    run_task.loadCheckpoint(path)

def main(cfg=None):
    parser = argparse.ArgumentParser(description="Evaluate PoseKitModel")
    parser.add_argument("--cfg", default=None, help="Path to config TOML")
    parser.add_argument("--weights", default=None, help="Path to .pth weights/checkpoint")
    args = parser.parse_args()

    if cfg is None:
        cfg = load_config(args.cfg)

    init(cfg)

    model = PoseKitModel(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')

    data = Data(cfg)
    data_loader = data.getEvalDataloader()

    run_task = Task(cfg, model)

    weights_path = _resolve_weights_path(cfg, args.weights)
    _load_weights(run_task, weights_path)
    run_task.evaluate(data_loader)

if __name__ == '__main__':
    main()
