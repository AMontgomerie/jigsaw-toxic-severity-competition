import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Mapping


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
            "labels": torch.tensor(item.target, dtype=torch.float32),
        }