import argparse
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import os

from utils import AverageMeter, set_seed
from dataset import ToxicDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--valid_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--dataloader_workers", type=int, default=2)
    parser.add_argument("--checkpoint", type=str, default="roberta-base")
    parser.add_argument("--seed", type=int, default=666)
    return parser.parse_args()


class Trainer:
    def __init__(
        self,
        fold: int,
        checkpoint: str,
        epochs: int,
        learning_rate: float,
        train_set: pd.DataFrame,
        valid_set: pd.DataFrame,
        train_batch_size: int,
        valid_batch_size: int,
        dataloader_workers: int,
        save_dir: str,
    ) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=1
        )
        self.model = self.model.to("cuda")
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=dataloader_workers,
        )
        self.valid_loader = DataLoader(
            valid_set,
            batch_size=valid_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=dataloader_workers,
        )
        self.train_loss = AverageMeter()
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.save_path = os.path.join(
            save_dir, f"{checkpoint.replace('/', '_')}_{fold}.bin"
        )
        self.best_valid_loss = float("inf")

    def train(self) -> None:
        for epoch in (1, self.epochs + 1):
            self.model.train()
            self.train_loss.reset()
            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for data in self.train_loader:
                    self.optimizer.zero_grad()
                    data = {k: v.to("cuda") for k, v in data.items()}
                    output = self.model(**data)
                    loss = output.loss
                    loss.backward()
                    self.optimizer.step()
                    self.train_loss.update(loss.item(), self.train_batch_size)
                    tepoch.set_postfix({"train_loss": self.train_loss.avg})
                    tepoch.update(1)
            valid_loss = self.evaluate()
            if valid_loss < self.best_valid_loss:
                print(
                    f"Valid loss decreased from {self.best_valid_loss} to {valid_loss}. Saving."
                )
                torch.save(self.model.state_dict(), self.save_path)
                self.best_valid_loss = valid_loss
            else:
                print(f"{valid_loss} is not an improvement.")

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        valid_loss = AverageMeter()
        with tqdm(total=len(self.valid_loader), unit="batches") as tepoch:
            tepoch.set_description("validation")
            for data in self.valid_loader:
                output = self.model(**data)
                loss = output.loss
                valid_loss.update(loss.item(), self.valid_batch_size)
                tepoch.set_postfix({"valid_loss": valid_loss.avg})
                tepoch.update(1)
        return valid_loss.avg


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    data = pd.read_csv(args.train_path)
    train_data = data.loc[data.fold != args.fold].reset_index(drop=True)
    valid_data = data.loc[data.fold == args.fold].reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    train_set = ToxicDataset(train_data, tokenizer)
    valid_set = ToxicDataset(valid_data, tokenizer)
    trainer = Trainer(
        fold=args.fold,
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_set=train_set,
        valid_set=valid_set,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        dataloader_workers=args.dataloader_workers,
        save_dir=args.save_dir,
    )
    trainer.train()