from lib import init, Data, PoseKitModel, Task
from config_loader import load_config

def main(cfg=None):
    if cfg is None:
        cfg = load_config()

    init(cfg)
    model = PoseKitModel(num_classes=cfg["num_classes"],
                    width_mult=cfg["width_mult"],
                    mode='train')
    data = Data(cfg)
    train_loader, val_loader = data.getTrainValDataloader()
    run_task = Task(cfg, model)
    run_task.train(train_loader, val_loader)

if __name__ == '__main__':
    main()
