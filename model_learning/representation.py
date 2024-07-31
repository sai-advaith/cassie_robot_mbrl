
# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

# Python imports
import math

# Representation Learning class
class Representation(nn.Module):
    """
    Projecting state action vector and adding nonlinearity
    """
    def __init__(self, input_size, representation_size, hidden_size1,
                 hidden_size2, dropout):
        super(Representation, self).__init__()

        # Parameters for input, hidden, and output
        self.output_size = representation_size
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        # Fully connected input -> hidden1
        self.fc1 = nn.Linear(self.input_size, self.hidden_size1)
        self.dropout1 = nn.Dropout(dropout)

        # Fully connected hidden1 -> hidden2
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.dropout2 = nn.Dropout(dropout)

        # Fully connected hidden2 -> output
        self.fc3 = nn.Linear(self.hidden_size2, self.output_size)

        # Layer norm for the output
        self.layer_norm = nn.LayerNorm(self.output_size)

    def action_embedding(self, x):
        """
        Adding sinusoidal embedding to allow gradient correction
        """
        embedding = torch.cat(([torch.sin(math.pi * x),
                                torch.cos(math.pi * x),
                                torch.sin(2 * math.pi * x),
                                torch.cos(2 * math.pi * x)]),
                                dim = 2)
        return embedding

    def forward(self, x):
        """
        Forward propagate tensor through multi layer perceptron
        """
        # Propagate input through input -> hidden 1
        x = self.action_embedding(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Hidden layer 2
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Pass through last fully connected layer
        x = self.fc3(x)

        # Return output of fully connected layer
        return self.layer_norm(x)
