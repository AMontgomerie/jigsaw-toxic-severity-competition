import argparse
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Mapping
from tqdm import tqdm

from utils import AverageMeter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--valid_path", type=str, default="data/valid.csv")
    parser.add_argument("--test_path", type=str, default="data/comments_to_score.csv")
    parser.add_argument("--save_path", type=str, default="./model.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--valid_batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--dataloader_workers", type=int, default=2)
    parser.add_argument("--checkpoint", type=str, default="roberta-base")
    return parser.parse_args()


class ToxicDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: AutoTokenizer) -> None:
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Mapping[torch.Tensor, torch.Tensor]:
        item = self.data.loc[index]
        encoded_item = self.tokenizer(
            item.text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded_item["input_ids"].squeeze(),
            "attention_mask": encoded_item["attention_mask"].squeeze(),
            "target": torch.tensor(item.target),
        }


class Trainer:
    def __init__(
        self,
        checkpoint: str,
        epochs: int,
        learning_rate: float,
        train_set: pd.DataFrame,
        valid_set: pd.DataFrame,
        train_batch_size: int,
        valid_batch_size: int,
        dataloader_workers: int,
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
    data = pd.read_csv(args.train_path)
    train_data = data[data.fold != 0]
    valid_data = data[data.fold == 0]
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    train_set = ToxicDataset(train_data, tokenizer)
    valid_set = ToxicDataset(valid_data, tokenizer)
    trainer = Trainer(
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_set=args.train_set,
        valid_set=args.valid_set,
        train_batch_size=args.train_batch_size,
        valid_batch_size=args.valid_batch_size,
        dataloader_workers=args.dataloader_workers,
    )
    trainer.train()