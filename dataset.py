import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Mapping


class ToxicDataset(Dataset):
    def __init__(
        self,
        texts: pd.Series,
        tokenizer: AutoTokenizer,
        max_length: int,
        labels: pd.Series = None,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Mapping[torch.Tensor, torch.Tensor]:
        text = self.texts.loc[index]
        encoded_item = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {
            "input_ids": encoded_item["input_ids"].squeeze(),
            "attention_mask": encoded_item["attention_mask"].squeeze(),
        }
        if self.labels:
            label = self.labels.loc[index]
            inputs["labels"] = torch.tensor(label, dtype=torch.float32)
        return inputs
