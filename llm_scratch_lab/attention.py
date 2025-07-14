import torch.nn as nn
import torch

class SelfAttention(nn.Module):
    def __init__(self, d_input, d_output):
        super().__init__()
        self.W_query = nn.Linear(torch.rand(d_input, d_output))
        self.W_key = nn.Linear(torch.rand(d_input, d_output))
        self.W_value = nn.Linear(torch.rand(d_input, d_output))

    def forward(self, input_tensor):
        """
        Forward pass for self-attention mechanism.
        
        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_input).
            This input is the sum of token and position embeddings.
        Returns:
            torch.Tensor: Context vector of shape (batch_size, seq_length, d_output).
        """
        keys = input_tensor @ self.W_key
        queries = input_tensor @ self.W_query
        values = input_tensor @ self.W_value
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vector = attention_weights @ values
        return context_vector
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_input, d_output, num_heads):
        super().__init__()
        pass

    def forward(self, x):
        pass