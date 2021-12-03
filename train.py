import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Mapping
from tqdm import tqdm

from utils import AverageMeter


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
    data = pd.read_csv("../data/train.csv")
    epochs = 1
    checkpoint = "roberta-base"
    train_batch_size = 32
    valid_batch_size = 128
    dataloader_workers = 2
    learning_rate = 1e-5
    train_data = data[data.fold != 0]
    valid_data = data[data.fold == 0]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    train_set = ToxicDataset(train_data, tokenizer)
    valid_set = ToxicDataset(valid_data, tokenizer)
    trainer = Trainer(
        checkpoint=checkpoint,
        epochs=epochs,
        learning_rate=learning_rate,
        train_set=train_set,
        valid_set=valid_set,
        train_batch_size=train_batch_size,
        valid_batch_size=valid_batch_size,
        dataloader_workers=dataloader_workers,
    )
    trainer.train()