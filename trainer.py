import torch
from torch.nn import MarginRankingLoss
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification
from tqdm import tqdm
import os
import numpy as np
from typing import Mapping, List

from utils import AverageMeter
from dataset import ToxicDataset, PairedToxicDataset


class Trainer:
    def __init__(
        self,
        fold: int,
        checkpoint: str,
        epochs: int,
        learning_rate: float,
        train_set: Dataset,
        valid_set: Dataset,
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
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.train_loss.reset()
            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for data in self.train_loader:
                    self.optimizer.zero_grad()
                    data = self._to_cuda(data)
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
                data = {k: v.to("cuda") for k, v in data.items()}
                output = self.model(**data)
                loss = output.loss
                valid_loss.update(loss.item(), self.valid_batch_size)
                tepoch.set_postfix({"valid_loss": valid_loss.avg})
                tepoch.update(1)
        return valid_loss.avg

    def _to_cuda(self, data: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        return {k: v.to("cuda") for k, v in data.items()}


class PairedTrainer(Trainer):
    def __init__(
        self,
        fold: int,
        checkpoint: str,
        epochs: int,
        learning_rate: float,
        train_set: PairedToxicDataset,
        less_toxic_valid_set: ToxicDataset,
        more_toxic_valid_set: ToxicDataset,
        train_batch_size: int,
        valid_batch_size: int,
        dataloader_workers: int,
        save_dir: str,
    ) -> None:
        super().__init__(
            fold,
            checkpoint,
            epochs,
            learning_rate,
            train_set,
            None,
            train_batch_size,
            valid_batch_size,
            dataloader_workers,
            save_dir,
        )
        self.less_toxic_valid_loader = DataLoader(
            less_toxic_valid_set,
            batch_size=valid_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=dataloader_workers,
        )
        self.more_toxic_valid_loader = DataLoader(
            more_toxic_valid_set,
            batch_size=valid_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=dataloader_workers,
        )
        self.loss_fn = MarginRankingLoss()
        self.best_valid_score = 0

    def train(self) -> None:
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.train_loss.reset()
            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for less_toxic_data, more_toxic_data, target in self.train_loader:
                    self.optimizer.zero_grad()
                    less_toxic_data = self._to_cuda(less_toxic_data)
                    more_toxic_data = self._to_cuda(more_toxic_data)
                    target = target.to("cuda")
                    less_toxic_output = self.model(**less_toxic_data)
                    more_toxic_output = self.model(**more_toxic_data)
                    loss = self.loss_fn(
                        less_toxic_output.logits, more_toxic_output.logits, target
                    )
                    loss.backward()
                    self.optimizer.step()
                    self.train_loss.update(loss.item(), self.train_batch_size)
                    tepoch.set_postfix({"train_loss": self.train_loss.avg})
                    tepoch.update(1)
            valid_score = self.evaluate()
            if valid_score > self.best_valid_score:
                print(
                    f"Valid score increased from {self.best_valid_score} to {valid_score}. Saving."
                )
                torch.save(self.model.state_dict(), self.save_path)
                self.best_valid_score = valid_score
            else:
                print(f"{valid_score} is not an improvement.")

    def evaluate(self) -> float:
        less_toxic_preds = self._predict(self.less_toxic_valid_loader, f"less toxic")
        more_toxic_preds = self._predict(self.more_toxic_valid_loader, f"more toxic")
        mean_less_toxic = np.mean(less_toxic_preds, axis=0)
        mean_more_toxic = np.mean(more_toxic_preds, axis=0)
        return sum(mean_less_toxic < mean_more_toxic) / len(
            self.less_toxic_valid_loader
        )

    @torch.no_grad()
    def _predict(
        self,
        dataloader: DataLoader,
        name: str,
    ) -> List[float]:
        self.model.eval()
        predictions = []
        with tqdm(total=len(dataloader), unit="batches") as tepoch:
            tepoch.set_description(name)
            for data in dataloader:
                data = self._to_cuda(data)
                output = self.model(**data)
                predictions += list(output.logits.squeeze().cpu().numpy())
                tepoch.update(1)
        return predictions
