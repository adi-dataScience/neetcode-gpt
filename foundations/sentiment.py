import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.embeddings_layer = nn.Embedding(vocabulary_size, 16)
        self.linear_layer = nn.Linear(16,1)
        self.sigmoid_layer = nn.Sigmoid()

        pass

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        embeddings = self.embeddings_layer(x)
        averaged = torch.mean(embeddings, dim = 1)
        projected = self.linear_layer(averaged)
        sigmoid = self.sigmoid_layer(projected)
        
        return torch.round(sigmoid, decimals = 4)
