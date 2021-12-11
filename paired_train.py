import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

from utils import set_seed
from dataset import ToxicDataset, PairedToxicDataset
from trainer import PairedTrainer
import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group_id", type=int, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--valid_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--dataloader_workers", type=int, default=2)
    parser.add_argument("--checkpoint", type=str, default="roberta-base")
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--scheduler", type=str, default="constant")
    parser.add_argument("--warmup", type=float, default=0)
    parser.add_argument("--loss_margin", type=float, default=0)
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--use_extra_data", dest="use_extra_data", action="store_true")
    parser.add_argument(
        "--extra_data_dir", type=str, default="data/extra_training_data"
    )
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = vars(args)
    if config["use_extra_data"]:
        extra_files = [
            os.path.join(config["extra_data_dir"], f)
            for f in os.listdir(config["extra_data_dir"])
        ]
        config["extra_files"] = extra_files
    wandb.login()
    with wandb.init(
        project="jigsaw-paired-train",
        group=str(args.group_id),
        name=f"{args.group_id}-{args.checkpoint}-fold-{args.fold}",
        config=config,
    ):
        config = wandb.config
        set_seed(config.seed)
        data = pd.read_csv(config.train_path)
        train_data = data.loc[data.fold != config.fold].reset_index(drop=True)
        valid_data = data.loc[data.fold == config.fold].reset_index(drop=True)
        if config.use_extra_data:
            extra_data = pd.concat([pd.read_csv(f) for f in extra_files])
            train_data = pd.concat([train_data, extra_data]).reset_index(drop=True)
        tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
        train_set = PairedToxicDataset(
            train_data.less_toxic, train_data.more_toxic, tokenizer, config.max_length
        )
        less_toxic_valid_set = ToxicDataset(
            valid_data.less_toxic, tokenizer, config.max_length
        )
        more_toxic_valid_set = ToxicDataset(
            valid_data.more_toxic, tokenizer, config.max_length
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            config.checkpoint, num_labels=1
        )
        trainer = PairedTrainer(
            fold=config.fold,
            model_name=config.checkpoint,
            model=model,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            train_set=train_set,
            less_toxic_valid_set=less_toxic_valid_set,
            more_toxic_valid_set=more_toxic_valid_set,
            train_batch_size=config.train_batch_size,
            valid_batch_size=config.valid_batch_size,
            dataloader_workers=config.dataloader_workers,
            save_dir=config.save_dir,
            scheduler=config.scheduler,
            warmup=config.warmup,
            loss_margin=config.loss_margin,
            early_stopping_patience=config.early_stopping_patience,
            log_interval=config.log_interval,
            weight_decay=config.weight_decay,
        )
        best_valid_score = trainer.train()
        print(f"Best valid score: {best_valid_score:.3f}")