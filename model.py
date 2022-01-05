import torch
import torch.nn as nn
from typing import Mapping


class ToxicLSTM(nn.Module):
    def __init__(self, embeddings: torch.Tensor, hidden_dim: int = 64) -> None:
        super(ToxicLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding.from_pretrained(embeddings)
        self.lstm = nn.LSTM(embeddings.shape[0], hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        embeddings = self.word_embeddings(sequence)
        lstm_out, _ = self.lstm(embeddings)
        return self.regressor(lstm_out)


def convert_regressor_to_binary(state_dict: Mapping) -> Mapping:
    state_dict["classifier.weight"] = torch.hstack(
        [state_dict["classifier.weight"]] * 2
    )
    state_dict["classifier.bias"] = torch.hstack([state_dict["classifier.bias"]] * 2)
    return state_dict