import torch.nn as nn
from LayerNormalization import LayerNormalization

class Decoder (nn.Module) :
    
    def __init__ (self, layers: nn.ModuleList) -> None :
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward (self, x, encoder_output, src_mask, tgt_mask) :
        for layer in self.layers :
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)