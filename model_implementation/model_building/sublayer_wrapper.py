# This file contains the implementation of the SubLayerWrapper class. This class is a wrapper around the
# MultiHeadedAttention and PositionwiseFeedForward classes. It applies the operation on the input, applies
# dropout, adds the input back to the transformed input, does normalization and returns the output. Please
# refer to 'step_12_encoder.ipynb' (add link to the notebook) notebook for a detailed explanation of
# how this class works.


from torch import nn, Tensor
from typing import Callable

# Notice that dropout and layer_norm are the child modules (part of BackPropagation) of the SubLayerWrapper 
# class. However, 'sublayer' (argument to the forward function) is not a child module of the SubLayerWrapper 
# class. It is passed as an argument to the forward method of the SubLayerWrapper class.
class SubLayerWrapper(nn.Module):
    def __init__(self, d_model: int, dropout_prob: float):
        """This class is a wrapper around the MultiHeadedAttention and PositionwiseFeedForward classes.

        Args:
            d_model (int): Dimension of the vectors used in the Attention model.
            dropout_prob (float): probability with which nodes can be dropped.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob, inplace=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input: Tensor, sublayer: Callable[[Tensor], Tensor]) -> Tensor:
        """It applies the operation on the input, applies dropout, adds the input back to the transformed 
           input, does normalization and returns the output.

        Args:
            input (Tensor): Input to be transformer by the sublayer.
                            shape: [batch_size, seq_len, d_model]
            sublayer (Callable): sublayer is a callable that takes a tensor as input and returns a tensor 
                                 as output. Could be either a lambda function that calls MultiHeadedAttention 
                                 or a direct nn.Module which is PositionwiseFeedForward in this case.
            
        Returns:
            Tensor: Output of the sublayer transformation.
                    shape: [batch_size, seq_len, d_model]
        """
        return input + self.dropout(sublayer(self.layer_norm(input)))