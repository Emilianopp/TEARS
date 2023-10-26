import torch
import torch.nn as nn
import rff


class decoderMLP(nn.Module):
    def __init__(self, input_dim, num_layers, output_dim, dropout=0.25):
        super(decoderMLP, self).__init__()

        # Calculate the reduction factor to evenly decrease dimensionality

        
        reduction_factor = (input_dim - (output_dim * 2)) // (num_layers - 1)
        
        # Initialize a list to hold the layers of the MLP
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, input_dim - reduction_factor))
        layers.append(nn.ReLU())  # You can choose a different activation function if needed
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers with residual connections
        for _ in range(num_layers - 2):  # Subtract 2 for the input and output layers
            residual = nn.Linear(input_dim - reduction_factor, input_dim - 2 * reduction_factor)
            layers.append(residual)
            layers.append(nn.ReLU())  # You can choose a different activation function if needed
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Identity())  # Identity mapping for residual connection
            input_dim -= reduction_factor
        
        # Output layer
        layers.append(nn.Linear(input_dim - reduction_factor, output_dim))
        layers.append(nn.Dropout(dropout))
        
        # Define the MLP as a Sequential model
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):

    
        return self.mlp(x)
    def mse_loss(self, pred, target):
       
        return torch.mean(torch.mean((pred - target)**2, dim=0))