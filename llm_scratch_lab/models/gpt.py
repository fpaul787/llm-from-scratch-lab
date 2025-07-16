import torch
import torch.nn as nn


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.__token_embedding__ = nn.Embedding(self.config["vocabulary_size"], self.config["embedding_dimension"]) # Token embedding layer
        self.__position_embedding__ = nn.Embedding(self.config["context_length"], self.config["embedding_dimension"]) # Position embedding layer
        # Dropout is optional

        # Initialize transformer blocks
        self.transformer_blocks = None


    
    def forward(self, input_tensor):
        """
        Forward pass for the GPT model.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_length) containing token indices.
        Returns:
            TBD
        """
        batch_size, seq_length = input_tensor.shape

        # Retrieves the meanings of the input tokens (words)
        token_embeddings = self.__token_embedding__(input_tensor) # Token embedding lookup.

        # Retrieves the positions of the input tokens
        position_embeddings = self.__position_embedding__(torch.arange(seq_length, device=input_tensor.device)) # Position embedding lookup
        
        x = token_embeddings + position_embeddings # Combine token and position embeddings
        pass