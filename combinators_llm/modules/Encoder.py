import torch.nn as nn
from combinators_llm.modules.LayerNormalization import LayerNormalization

class Encoder (nn.Module) :
    
    def __init__(self, layers: nn.ModuleList) -> None :
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, mask) :
        for layer in self.layers :
            x = layer(x, mask)
        return self.norm(x)