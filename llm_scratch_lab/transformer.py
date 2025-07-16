import torch.nn as nn

from llm_scratch_lab.attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_input=config["embedding_dimension"],
            d_output=config["embedding_dimension"],
            num_heads=config["num_heads"],
            context_length=config["context_length"]
        )

    def forward(self, x):
        pass