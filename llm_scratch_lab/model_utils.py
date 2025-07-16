import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    """Layer normalization for stabilizing training."""
    def __init__(self, embedding_dimension, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(embedding_dimension))
        self.bias = nn.Parameter(torch.zeros(embedding_dimension))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * normalized_x + self.bias