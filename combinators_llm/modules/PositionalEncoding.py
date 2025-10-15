import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)  # (seq_len, d_model)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # numerical stability

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(
            position
            * div_term[: d_model // 2 if d_model % 2 == 0 else (d_model // 2 + 1)]
        )

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)  # type: ignore
        return self.dropout(x)
