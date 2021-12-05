import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Mapping, Tuple


class ToxicDataset(Dataset):
    def __init__(
        self,
        texts: pd.Series,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        labels: pd.Series = None,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> Mapping[str, torch.Tensor]:
        text = self.texts.loc[index]
        inputs = self._encode(text)
        if self.labels is not None:
            label = self.labels.loc[index]
            inputs["labels"] = torch.tensor(label, dtype=torch.float32)
        return inputs

    def _encode(self, text: str) -> Mapping[str, torch.Tensor]:
        encoded_item = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded_item["input_ids"].squeeze(),
            "attention_mask": encoded_item["attention_mask"].squeeze(),
        }


class PairedToxicDataset(ToxicDataset):
    def __init__(
        self,
        less_toxic: pd.Series,
        more_toxic: pd.Series,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ) -> None:
        super().__init__(None, tokenizer, max_length)
        self.less_toxic = less_toxic
        self.more_toxic = more_toxic

    def __len__(self) -> int:
        return len(self.less_toxic)

    def __getitem__(
        self, index: int
    ) -> Tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
        less_toxic_text = self.less_toxic.loc[index]
        more_toxic_text = self.more_toxic.loc[index]
        less_toxic_inputs = self._encode(less_toxic_text)
        more_toxic_inputs = self._encode(more_toxic_text)
        return less_toxic_inputs, more_toxic_inputs
