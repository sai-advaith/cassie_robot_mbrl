# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

# Python imports
from model_learning.positional_encoding import PositionalEncoder
import math

# Transformer Model Class
class Transformer(nn.Module):
    """
    Encoder-Decoder based transformer model.
    Task: predict the difference in states given a context of transitions
    Constraints: Include a mask which prevent interaction with future steps
    """
    def __init__(self, representation_size, output_size, d_model, n_heads,
                 num_encoders, num_decoders, history, dropout):
        """
        representation_size: size of input to the transformer model
        output_size: state difference prediction size
        d_model: transformer encoding size
        n_heads: for multi head attention
        num_encoders: encoders to perform multihead attention
        H: context for predicting the next state
        dropout: dropout probability for better generalization
        """
        super(Transformer, self).__init__()

        # Input to the encoder
        self.d_model = d_model
        self.encoder_input = nn.Linear(representation_size, self.d_model)

        # Pass through positional encoder
        self.encoder_pe = PositionalEncoder(dropout, history, self.d_model)
        self.decoder_pe = PositionalEncoder(dropout, history - 1, self.d_model)

        # Encoder layers (self-attention)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model,
                                                   nhead=n_heads,
                                                   batch_first=True,
                                                   dropout=dropout)

        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=num_encoders)

        # Decoder Input Layer
        self.decoder_input = nn.Linear(output_size, self.d_model)

        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(self.d_model, n_heads,
                                                   batch_first=True,
                                                   dropout=dropout)

        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer,
                                             num_layers=num_decoders)

        # Output layer
        self.linear_mapping = nn.Linear(d_model, output_size)

    def generate_square_subsequent_mask(self, seq_length):
        """
        Square mask for self-attention of size seq_length x seq_length 
        """
        mask = torch.triu(torch.ones((seq_length, seq_length))).transpose(0, 1)
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        # Apply relevant conversions
        mask = mask.double()
        if torch.cuda.is_available():
            mask = mask.cuda()

        return mask

    def generate_memory_mask(self, output_length, input_length):
        """
        Mask for self-attention between decoder input and encoder output
        """
        mask = (torch.triu(torch.ones(input_length, output_length))).transpose(0, 1)
        memory_mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        # Apply conversion
        memory_mask = memory_mask.double()
        if torch.cuda.is_available():
            memory_mask = memory_mask.cuda()
        return memory_mask

    def forward(self, x, target, src_mask = None, target_mask = None, memory_mask = None):
        """
        x: soruce sequence i.e. original state action transition sequence
        target: output sequence for teacher forcing
        src_mask: no data leak to future for x during self-attention
        target_mask: no data leak to future for target during self-attention
        memory_mask: no data leak from encoded x and target

        Example:
        X = ((s_1, a_1), (s_2, a_2), ... (s_H, a_H))
        target = ((s_2 - s_1), (s_3 - s_2), .... (s_H - s_{H-1}))
        decoder output = ((s_3 - s_2), (s_4 - s_3), .... (s_{H+1} - s_{H}))
        """
        # Embed target input to the decoder
        x = self.encoder_input(x)# * math.sqrt(self.d_model)

        # Positional encoding
        x = self.encoder_pe(x)
        encoder_output = self.encoder(x, mask=src_mask)

        # Embed target input to the decoder
        decoder_output = self.decoder_input(target)# * math.sqrt(self.d_model)

        # Positional encoding
        decoder_output = self.decoder_pe(decoder_output)
        decoder_output = self.decoder(tgt=decoder_output,
                                      memory=encoder_output,
                                      tgt_mask=target_mask,
                                      memory_mask=memory_mask)

        # Map back to our dimensionality
        prediction = self.linear_mapping(decoder_output)
        return prediction[:, -1:, :]
