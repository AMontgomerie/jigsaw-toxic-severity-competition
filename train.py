import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import set_seed
from dataset import ToxicDataset
from trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--scheduler", type=str, default="constant")
    parser.add_argument("--warmup", type=float, default=0)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    data = pd.read_csv(args.train_path)
    train_data = data.loc[data.fold != args.fold].reset_index(drop=True)
    valid_data = data.loc[data.fold == args.fold].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.checkpoint, num_labels=1
    )
    train_set = ToxicDataset(
        train_data.text, tokenizer, args.max_length, train_data.target
    )
    valid_set = ToxicDataset(
        valid_data.text, tokenizer, args.max_length, valid_data.target
    )
    trainer = Trainer(
        fold=args.fold,
        model=model,
        model_name=args.checkpoint,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_set=train_set,
        valid_set=valid_set,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        dataloader_workers=args.dataloader_workers,
        save_dir=args.save_dir,
        scheduler=args.scheduler,
        warmup=args.warmup,
        early_stopping_patience=0,
        log_interval=args.log_interval,
        weight_decay=args.weight_decay,
        accumulation_steps=args.accumulation_steps,
    )
    trainer.train()