import torch
import torch.nn as nn
from torch.nn import MarginRankingLoss
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
from transformers import AutoModelForSequenceClassification, get_scheduler
from tqdm import tqdm
import os
import numpy as np
from typing import Mapping, List
import wandb

from utils import AverageMeter
from dataset import ToxicDataset, PairedToxicDataset


class Trainer:
    def __init__(
        self,
        fold: int,
        model_name: str,
        model: nn.Module,
        epochs: int,
        learning_rate: float,
        train_set: Dataset,
        valid_set: Dataset,
        train_batch_size: int,
        valid_batch_size: int,
        dataloader_workers: int,
        save_dir: str,
        scheduler: str,
        warmup: float,
        early_stopping_patience: int,
        log_interval: int,
        weight_decay: float,
        accumulation_steps: int,
    ) -> None:
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
        self.model = model
        self.model = self.model.to("cuda")
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
        self.best_valid_score = float("inf")
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0
        self.log_interval = log_interval

    def train(self) -> None:
        global_step = 1
        self.optimizer.zero_grad(set_to_none=True)
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.train_loss.reset()
            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for data in self.train_loader:
                    data = self._to_cuda(data)
                    output = self.model(**data)
                    loss = output.loss
                    loss = loss / self.accumulation_steps
                    loss.backward()
                    if global_step % self.accumulation_steps == 0:
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                    self.train_loss.update(loss.item(), self.train_batch_size)
                    tepoch.set_postfix({"train_loss": self.train_loss.avg})
                    tepoch.update(1)
                    global_step += 1
            valid_score = self.evaluate()
            terminate = self._on_epoch_end(
                valid_score < self.best_valid_score, valid_score
            )
            if terminate:
                return

    def _on_epoch_end(self, score_improved: bool, valid_score: float) -> None:
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
        model_name: str,
        model: nn.Module,
        epochs: int,
        learning_rate: float,
        train_set: PairedToxicDataset,
        less_toxic_valid_set: ToxicDataset,
        more_toxic_valid_set: ToxicDataset,
        train_batch_size: int,
        valid_batch_size: int,
        dataloader_workers: int,
        save_dir: str,
        scheduler: str,
        warmup: float,
        early_stopping_patience: int,
        loss_margin: float,
        log_interval: int,
        weight_decay: float,
        validation_steps: int,
        accumulation_steps: int,
    ) -> None:
        super().__init__(
            fold,
            model_name,
            model,
            epochs,
            learning_rate,
            train_set,
            None,
            train_batch_size,
            valid_batch_size,
            dataloader_workers,
            save_dir,
            scheduler,
            warmup,
            early_stopping_patience,
            log_interval,
            weight_decay,
            accumulation_steps,
        )
        on_fail = "validation dataset lengths don't match!"
        assert len(less_toxic_valid_set) == len(more_toxic_valid_set), on_fail
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
        self.loss_fn = MarginRankingLoss(loss_margin)
        self.best_valid_score = 0
        self.wandb_train_loss = AverageMeter()
        self.scaler = amp.GradScaler()
        self.validation_steps = validation_steps

    def train(self) -> float:
        wandb.watch(self.model, self.loss_fn, log="all", log_freq=self.log_interval)
        wandb.log({"valid_score": 0})
        global_step = 1
        self.optimizer.zero_grad(set_to_none=True)
        print("validation steps", self.validation_steps)
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.train_loss.reset()
            self.wandb_train_loss.reset()
            with tqdm(total=len(self.train_loader), unit="batches") as tepoch:
                tepoch.set_description(f"epoch {epoch}")
                for epoch_step, (less_toxic_data, more_toxic_data, target) in enumerate(
                    self.train_loader
                ):
                    print(epoch_step)
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
                        and epoch_step + 1 % self.validation_steps == 0
                    ):
                        valid_score = self.evaluate(use_tqdm=False)
                        self.model.train()
                        wandb.log({"valid_score": valid_score})
                        terminate = self._on_epoch_end(
                            valid_score > self.best_valid_score, valid_score
                        )
                        if terminate:
                            return self.best_valid_score

                    global_step += 1

            if self.validation_steps is None:
                valid_score = self.evaluate()
                wandb.log({"valid_score": valid_score})
                terminate = self._on_epoch_end(
                    valid_score > self.best_valid_score, valid_score
                )
                if terminate:
                    return self.best_valid_score

        return self.best_valid_score

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
