from llm_scratch_lab.tokenizer import Tokenizer
from llm_scratch_lab.models.gpt import GPTModel

import torch

# Define the configuration for the GPT model
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

class GPTLab:
    """A class representing a GPT Lab environment."""

    def __init__(self):
        self.__tokens__ = None
        self.__token_tensor__ = None
        self.config = GPT_CONFIG_124M
        self.tokenizer = Tokenizer(encoding_name="gpt2")
        self.model = GPTModel(self.config)
    
    def __tensorize_tokens__(self, tokens, with_batch_dim=True):
        """Converts tokens to a tensor format."""
        tensor = torch.tensor(tokens, dtype=torch.long)
        if with_batch_dim:
            tensor = tensor.unsqueeze(0)  # Add batch dimension
        return tensor

    def run(self, text):
        print("Running GPT Lab...")
        print("Tokenizing the input text...")
        # Tokenize the input text
        self.__tokens__ = self.tokenizer.tokenize(text)

        print("Converting tokens to PyTorch tensor...")
        # Tensorize the tokens
        self.__token_tensor__ = self.__tensorize_tokens__(self.__tokens__)