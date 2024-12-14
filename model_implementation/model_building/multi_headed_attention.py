# This file implements the Multi-Headed Attention layer that is used in the Transformer model.
# Refer to 'step_9_multi_headed_attention.ipynb' (link to the notebook) notebook for a detailed 
# explanation of each line of code in this file.
# The transformation of the input in the Multi-Headed Attention layer is shown visually in this
# pdf - 'Data/Resources/Input_Transformation_Multi_Headed_Attention.pdf' (ADD LINK TO THE PDF). 
# Please refer to this pdf if the tensor manipulations in the forward function seem confusing.

from model_implementation.utils.helpers import clone_module
from model_implementation.utils.logger import get_logger
from model_implementation.utils.constants import DROPOUT_PROB
from torch import nn, Tensor
from typing import Optional

import math
import torch

# Get the logger for this file.
logger = get_logger(__name__)

def construct_attention_heads(queries: Tensor, 
                              keys: Tensor, 
                              values: Tensor, 
                              mask: Optional[Tensor]=None, 
                              dropout_layer: Optional[nn.Module]=None) -> Tensor:
    """Calculates the attention scores for each token in the sequence with every other token in the sequence.
       Applies the mask if provided and then normalizes the scores using softmax. It then calculates the 
       attention heads for each token in the sequence.

    Args:
        queries (Tensor): [batch_size, num_heads, seq_len, d_k]
        keys (Tensor): [batch_size, num_heads, seq_len, d_k]
        values (Tensor): [batch_size, num_heads, seq_len, d_k]
        mask (Optional[Tensor]): [batch_size, 1, 1, src_seq_len] if the mask is for the source sequences.
                                 [batch_size, 1, tgt_seq_len, tgt_seq_len] if the mask is for the target sequences.
                                 Defaults to None.
        dropout_layer (Optional[nn.Module], optional): probability with which the values are dropped on dropout 
                                                       layer. Defaults to None.

    Returns:
        Tensor: Returns the attention heads.
                SHAPE: [batch_size, num_heads, seq_len, d_k]
    """
    # Size of the vectors for each token for each head in the sequence.
    d_k = queries.shape[-1]
    # Calculate the attention scores for each token in the sequence with every other token in the sequence.
    attention_scores = torch.matmul(queries, keys.transpose(dim0=2, dim1=3)) / math.sqrt(d_k)
    # Mask the attention scores if a mask is provided. Mask is used in two different ways:
    # 1) To prevent the model from attending to the padding tokens --> This applies for both src and tgt sequences.
    # 2) To prevent the model from attending to the future tokens in the sequence --> This applies only for tgt sequences.
    if mask is not None:
        # Please do not set the masked values to float('-inf') as it sometimes (not in everycase) causes softmax to return nan.
        attention_scores = attention_scores.masked_fill(mask == False, float('-1e9'))
    # Normalize the attention scores using softmax.
    attention_scores = attention_scores.softmax(dim=-1)
    # Apply dropout regularization to prevent overfitting problems.
    if dropout_layer is not None:
        attention_scores = dropout_layer(attention_scores)
    # The result of this matrix multiplication is the attention_heads.
    # Calculate the attention heads for each token in the sequence. The head for each token is calculated by
    # taking the weighted average (averaged by attention scores) of the values for all the tokens in the 
    # sequence for the token of interest. 
    return torch.matmul(attention_scores, values)



class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads: int, d_model: int, dropout_prob: float=DROPOUT_PROB):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        # We use dropout to prevent overfitting.
        self.dropout_layer = nn.Dropout(p=dropout_prob, inplace=False)
        # Creating the linear layers that generate queries, keys and values for each token in the sequence.
        # Also, creating an additional linear layer to generate the output of the Multi-Headed Attention from concatenated attention heads.
        self.linear_layers = clone_module(module=nn.Linear(in_features=d_model, out_features=d_model), num_clones=4)


    def forward(self, query_input: Tensor, key_input: Tensor, value_input: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        """Forward pass of the Multi-Headed Attention layer. 

        Args:
            query_input (Tensor): Input to be used for query creation.
                                  SHAPE: [batch_size, seq_len, d_model]
            key_input (Tensor): Input to be used for key creation.
                                SHAPE: [batch_size, seq_len, d_model]
            value_input (Tensor): Input to be used for value creation.
                                  SHAPE: [batch_size, seq_len, d_model]
            mask (Tensor): Mask to be applied to the attention scores. Default is None. Same mask will 
                           be applied to all the heads in the Multi-Headed Attention layer.
                           mask: [batch_size, 1, 1, src_seq_len] if the mask is for the source sequences.
                           mask: [batch_size, 1, tgt_seq_len, tgt_seq_len] if the mask is for the target sequences. 
                           Note that src_seq_len and tgt_seq_len are the number of tokens in the source and target sequences
                           respectively and they are likely different.

        Returns:
            Mutli-Headed Attention Output: Output of the Multi-Headed Attention layer. Generates one output vector 
                                           for each token in the sequence. Does this for each sequence in the batch.
                                           SHAPE: [batch_size, seq_len, d_model]
        """
        logger.debug("POINT 0 -- Inside the forward pass of the Multi-Headed Attention layer.")
        # Generates the queries, keys and values for each token in the sequence.
        # shape of queries, keys, values: [batch_size, seq_len, d_model]
        queries, keys, values = [linear_layer(input) for linear_layer, input in zip(self.linear_layers, (query_input, key_input, value_input))]
        batch_size = query_input.shape[0]
        # Using '-1' in the view function is to infer the size of the dimension from the original tensor. This is important because
        # the 'seq_len' for the keys, values comes from Encoder output (i.e., src sequences) and the 'seq_len' for the queries comes
        # from decoder input (i.e., tgt sequences) in source attention. The src_sequence size and tgt_sequence size are likely 
        # different and are being handled with common functionality here. So, we need to infer the size of the dimension from the 
        # original tensor instead of harcoding it from the query_input tensor. You can try it by hardcoding the seq_len (instead of setting it to -1) 
        # for keys and values and see the error you get to understand it better (I found out this issue after noticing the errors).
        # This separates the queries, keys and values for each head into a separate vector (thus a 4D tensor). The vectors for each 
        # token in all the heads are concatenated when they are created using the linear_layers above.
        # Shape for queries, keys, values after view: [batch_size, seq_len, num_heads, d_k]
        # Shape for queries, key, values after transpose: [batch_size, num_heads, seq_len, d_k]
        queries, keys, values = [data.view(batch_size, -1, self.num_heads, self.d_k).transpose(dim0=1, dim1=2) for data in (queries, keys, values)]
        # Calculate the attention heads for each token in the sequence.
        # attention_heads: [batch_size, num_heads, seq_len, d_k]
        attention_heads = construct_attention_heads(queries=queries, keys=keys, values=values, mask=mask, dropout_layer=self.dropout_layer)
        # Concatenate the attention heads for each token from all the heads.
        # attention_heads: [batch_size, seq_len, d_model]
        attention_heads = attention_heads.transpose(dim0=1, dim1=2).reshape(batch_size, -1, self.d_model)
        # Generate the output of the Multi-Headed Attention layer.
        return self.linear_layers[-1](attention_heads)