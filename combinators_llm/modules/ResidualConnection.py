import torch.nn as nn
from combinators_llm.modules.LayerNormalization import LayerNormalization

class ResidualConnection (nn.Module) :
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer) :
        return x + self.dropout(sublayer(self.norm(x)))