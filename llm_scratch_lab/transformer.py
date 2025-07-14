import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = None

    def forward(self, x):
        pass