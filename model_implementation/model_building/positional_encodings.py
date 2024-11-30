# This file implements the positional encodings. It takes in the encoding size, dropout probability and creates
# the positional encodings for the transformer model. Please refer to 'step_8_positional_encoding.ipynb' (link to the notebook) 
# notebook to understand the details about each step in the below code.

from model_implementation.utils.constants import MAX_INPUT_SEQUENCE_LENGTH
from torch import nn, Tensor

import math
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, encoding_size: int, dropout_prob: float, max_len: int=MAX_INPUT_SEQUENCE_LENGTH):
        """Creates the positional encodings.

        Args:
            encoding_size (int): Size of the positional encoding vector that represents the position of the token.
            dropout_prob (float): Probability of an element to be zeroed or dropped.
            max_len (int): Largest position for which the positional encoding vector is generated. Defaults to 
                           MAX_SEQUENCE_LENGTH (5000). By default, it generates positional encodings for the first 
                           MAX_SEQUENCE_LENGTH (5000) positions.
        """
        super().__init__()
        # Refer to step_7_drop_out.ipynb notebook (link to the notebook) to understand more about dropout.
        self.dropout = nn.Dropout(p=dropout_prob, inplace=False)
        # Compute the positional encodings in log space.
        positional_encoding = torch.zeros(size=(max_len, encoding_size), dtype=torch.float)
        positional_encoding_numerators = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        numerators_in_exponent = torch.arange(0, encoding_size, 2, dtype=torch.float)
        positional_encoding_denominators = torch.exp(numerators_in_exponent * (-math.log(10000.0) / encoding_size))
        positional_encoding[:, 0::2] = torch.sin(positional_encoding_numerators * positional_encoding_denominators)
        positional_encoding[:, 1::2] = torch.cos(positional_encoding_numerators * positional_encoding_denominators)
        # Refer to understanding_tensor_manipulations_part_1.ipynb notebook (add link to the notebook) to
        # understand more about unsqueeze operation in pytorch.
        # In transformer model, we receive 3D tensors as input to this module. Each 1D tensor
        # in the last dimension is an embedding for the token. Each 2D tensor is a sentence.
        # The entire 3D tensor is a batch of sentences. To work with 3D tensors in the forward
        # method, we convert the positional encoding to a 3D tensor.
        positional_encoding = positional_encoding.unsqueeze(0)
        # Refer to using_modules.ipynb (link to the notebook) to understand more about buffers in pytorch.
        # Essentially, This tells the module to not update the positional encoding tensor during the training. 
        # It is not a trainable parameter but it is still part of the state of the model.
        self.register_buffer('positional_encoding', positional_encoding)
    
    def forward(self, input: Tensor) -> Tensor:
        """Adds the positional encodings to the input tensor.
        Args:
            input (Tensor): The input tensor containing the embeddings of the tokens.
                            shape: [batch_size, sequence_length, d_model]

        Returns:
            Tensor: Input with the positional encodings added to it.
                    shape: [batch_size, sequence_length, d_model]
        """
        # Refer to understanding_tensor_manipulations_part_5.ipynb notebook (add link to the notebook) to 
        # understand more about broadcasting in python.
        # The input tensor is a 3D tensor of shape (batch_size, sequence_length, encoding_size).
        # We add (uses broadcasting) the positional encoding to the input tensor to get the final tensor.
        # positional_encoding: (1, max_len, encoding_size) --> (1, sequence_length, encoding_size) 
        #       -- Extracts the positional encodings for the sequence_length from the positional_encoding 
        #          tensor.
        # (batch_size, sequence_length, encoding_size) --> input
        # (batch_size, sequence_length, encoding_size) --> Resultant positional encoding tensor after broadcasting.
        # requires_grad_(False) is not needed since the positional encoding is already registered
        # as a Buffer and not a trainable parameter. It is just included for clarity.
        input = input + self.positional_encoding[:, :input.size(1)].requires_grad_(False)
        return self.dropout(input)
