# Pytorch imports
import torch
import torch.nn as nn
from torch.autograd import Variable

# Python imports
import math

# Positional Encoder Class
class PositionalEncoder(nn.Module):
    """
    Positional Encoder
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, dropout, H, d_model):
        super(PositionalEncoder, self).__init__()
        # Adding positional encoding based on maximum indices
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Positions
        position = torch.arange(0, H).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.Tensor([10000.0])) / d_model))

        # Create embedding
        self.positional_encoding = torch.zeros(H, d_model)
        if torch.cuda.is_available():
            self.positional_encoding = self.positional_encoding.cuda()
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)

        # Shape change to match x
        self.positional_encoding = self.positional_encoding.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        # Add to what must be encoded as per https://arxiv.org/pdf/1706.03762.pdf
        # Do not differentiate sinusoidal encoding
        x = x + Variable(self.positional_encoding.squeeze(1), requires_grad=False)
        return self.dropout(x)
