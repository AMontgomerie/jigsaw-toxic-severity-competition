import torch
import torch.nn as nn
from torch.nn import MarginRankingLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda import amp
from transformers import get_scheduler
from tqdm import tqdm
import os
import numpy as np
from typing import Mapping, List, Tuple, Union
import wandb

from utils import AverageMeter
from dataset import ToxicDataset, PairedToxicDataset


class Trainer:
    def __init__(
        self,
        accumulation_steps: int,
        dataloader_workers: int,
        early_stopping_patience: int,
        epochs: int,
        fold: int,
        learning_rate: float,
        less_toxic_valid_set: ToxicDataset,
        log_interval: int,
        loss_type: str,
        model_name: str,
        model: nn.Module,
        num_labels: int,
        more_toxic_valid_set: ToxicDataset,
        train_batch_size: int,
        train_set: ToxicDataset,
        save_dir: str,
        scheduler: str,
        valid_batch_size: int,
        validation_steps: int,
        warmup: float,
        weight_decay: float,
    ) -> None:
        self.train_loader = DataLoader(
            train_set,
            batch_size=train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=dataloader_workers,
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
        self.model = model
        self.model = self.model.to("cuda")
        self.loss_type = loss_type
        if self.loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif self.loss_type == "ce":
            self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.epochs = epochs
        self.accumulation_steps = accumulation_steps
        total_steps = (len(self.train_loader) // self.accumulation_steps) * self.epochs
        num_warmup_steps = round(total_steps * warmup)
        self.scheduler = get_scheduler(
            scheduler, self.optimizer, num_warmup_steps, total_steps
        )
        self.train_loss = AverageMeter()
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.save_path = os.path.join(
            save_dir, f"{model_name.replace('/', '_')}_{fold}.bin"
        )
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0
        self.log_interval = log_interval
        self.best_valid_score = 0
        self.scaler = amp.GradScaler()
        self.wandb_train_loss = AverageMeter()
        self.validation_steps = validation_steps
        self.num_labels = num_labels

    def train(self) -> float:
        wandb.watch(self.model, log="all", log_freq=self.log_interval)
        wandb.log({"valid_score": 0})
        global_step = 1
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.train_loss.reset()
            self.wandb_train_loss.reset()
            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for epoch_step, data in enumerate(self.train_loader):
                    if self.num_labels > 1:
                        data["labels"] = data["labels"].unsqueeze(1)
                    loss = self._model_fn(data)
                    self.scaler.scale(loss).backward()
                    if global_step % self.accumulation_steps == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    self.train_loss.update(loss.item(), self.train_batch_size)
                    self.wandb_train_loss.update(loss.item(), self.train_batch_size)
                    if global_step % self.log_interval == 0:
                        wandb.log(
                            {"epoch": epoch, "train_loss": self.wandb_train_loss.avg},
                            step=global_step,
                        )
                        self.wandb_train_loss.reset()
                    tepoch.set_postfix({"train_loss": self.train_loss.avg})
                    tepoch.update(1)

                    if (
                        self.validation_steps is not None
                        and (epoch_step + 1) % self.validation_steps == 0
                    ):
                        valid_score = self.evaluate(use_tqdm=False)
                        self.model.train()
                        wandb.log({"valid_score": valid_score})
                        terminate = self._on_eval(
                            valid_score > self.best_valid_score, valid_score
                        )
                        if terminate:
                            return self.best_valid_score

                    global_step += 1

            if self.validation_steps is None:
                valid_score = self.evaluate()
                wandb.log({"valid_score": valid_score})
                terminate = self._on_eval(
                    valid_score > self.best_valid_score, valid_score
                )
                if terminate:
                    return self.best_valid_score

        return self.best_valid_score

    def _model_fn(self, data: Mapping[str, torch.Tensor]) -> torch.Tensor:
        data = self._to_cuda(data)
        with amp.autocast():
            output = self.model(
                input_ids=data["input_ids"], attention_mask=data["attention_mask"]
            )
            loss = self._loss_fn(output.logits, data["labels"])
            loss = loss / self.accumulation_steps
        return loss

    def _loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "mse":
            loss = self.loss_fn(logits.squeeze(), labels)
        elif self.loss_type == "ce":
            predictions = torch.argmax(logits, dim=1).int()
            loss = self.loss_fn(predictions, labels.squeeze().int())
        else:
            raise NotImplementedError()
        return loss

    def _on_eval(self, score_improved: bool, valid_score: float) -> None:
        if score_improved:
            print(
                f"Valid score improved from {self.best_valid_score} to {valid_score}. Saving."
            )
            torch.save(self.model.state_dict(), self.save_path)
            self.best_valid_score = valid_score
            self.early_stopping_counter = 0
        else:
            if self.early_stopping_patience > 0:
                self.early_stopping_counter += 1
                print(
                    f"{valid_score} is not an improvement. "
                    f"Early stopping {self.early_stopping_counter}/{self.early_stopping_patience}"
                )
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print("Terminating.")
                    return True
            else:
                print(f"{valid_score} is not an improvement.")
        return False

    def _to_cuda(self, data: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        return {k: v.to("cuda") for k, v in data.items()}

    def evaluate(self, use_tqdm: bool = True) -> float:
        if use_tqdm:
            less_toxic_preds = self._predict_tqdm(
                self.less_toxic_valid_loader, f"valid (less toxic)"
            )
            more_toxic_preds = self._predict_tqdm(
                self.more_toxic_valid_loader, f"valid (more toxic)"
            )
        else:
            less_toxic_preds = self._predict(self.less_toxic_valid_loader)
            more_toxic_preds = self._predict(self.more_toxic_valid_loader)
        return sum(less_toxic_preds < more_toxic_preds) / len(less_toxic_preds)

    @torch.no_grad()
    def _predict_tqdm(
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
        return np.array(predictions)

    @torch.no_grad()
    def _predict(self, dataloader: DataLoader) -> List[float]:
        self.model.eval()
        predictions = []
        for data in dataloader:
            data = self._to_cuda(data)
            output = self.model(**data)
            predictions += list(output.logits.squeeze().cpu().numpy())
        return np.array(predictions)


class PairedTrainer(Trainer):
    def __init__(
        self,
        accumulation_steps: int,
        dataloader_workers: int,
        early_stopping_patience: int,
        epochs: int,
        fold: int,
        learning_rate: float,
        less_toxic_valid_set: ToxicDataset,
        log_interval: int,
        loss_margin: float,
        model_name: str,
        model: nn.Module,
        more_toxic_valid_set: ToxicDataset,
        train_batch_size: int,
        train_set: PairedToxicDataset,
        save_dir: str,
        scheduler: str,
        valid_batch_size: int,
        validation_steps: int,
        warmup: float,
        weight_decay: float,
    ) -> None:
        super().__init__(
            accumulation_steps,
            dataloader_workers,
            early_stopping_patience,
            epochs,
            fold,
            learning_rate,
            less_toxic_valid_set,
            log_interval,
            "ranking_loss",
            model_name,
            model,
            more_toxic_valid_set,
            train_batch_size,
            train_set,
            save_dir,
            scheduler,
            valid_batch_size,
            validation_steps,
            warmup,
            weight_decay,
        )
        self.loss_fn = MarginRankingLoss(loss_margin)

    def _model_fn(
        self, data: Tuple[Union[Mapping[str, torch.Tensor], torch.Tensor]]
    ) -> torch.Tensor:
        (less_toxic_data, more_toxic_data, target) = data
        less_toxic_data = self._to_cuda(less_toxic_data)
        more_toxic_data = self._to_cuda(more_toxic_data)
        target = target.to("cuda")
        with amp.autocast():
            less_toxic_output = self.model(**less_toxic_data)
            more_toxic_output = self.model(**more_toxic_data)
            loss = self.loss_fn(
                less_toxic_output.logits, more_toxic_output.logits, target
            )
            loss = loss / self.accumulation_steps
        return loss
