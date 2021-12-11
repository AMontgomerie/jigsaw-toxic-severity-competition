import torch
import torch.nn as nn


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