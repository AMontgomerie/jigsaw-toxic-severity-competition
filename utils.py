import argparse
import numpy as np
import torch
import os


class AverageMeter(object):
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_training_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default="roberta-base")
    parser.add_argument("--dataloader_workers", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument(
        "--extra_data_dir", type=str, default="data/extra_training_data"
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--group_id", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--loss_margin", type=float, default=0.5)
    parser.add_argument("--loss_type", type=str, default="mse")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_labels", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--scheduler", type=str, default="constant")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--use_extra_data", dest="use_extra_data", action="store_true")
    parser.add_argument("--valid_batch_size", type=int, default=128)
    parser.add_argument("--validation_steps", type=int, default=None)
    parser.add_argument("--valid_path", type=str, default=None)
    parser.add_argument("--warmup", type=float, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--weights_path", type=str, default=None)
    return parser.parse_args()