# This file implements the Encoder in the Transformer model. The Encoder is a stack of N=6 identical EncoderLayers.
# Refer to 'step_12_encoder.ipynb' (add link to the notebook) notebook for a detailed explanation of each line of
# code in this file.

from model_implementation.model_building.feed_forward_nn import FeedForwardNN
from model_implementation.model_building.multi_headed_attention import MultiHeadedAttention
from model_implementation.model_building.sublayer_wrapper import SubLayerWrapper
from model_implementation.utils.helpers import clone_module
from model_implementation.utils.logger import get_logger
from torch import nn, Tensor
from typing import Optional

logger = get_logger(__name__)

# The MultiHeadedAttention (self_attention here) and FeedForward modules are also common (common meaning they 
# have the same implementation and instantiation mechanism and not that they share weights) to the DecoderLayer.
# Hence, we create them in a common way at the top level and pass them as arguments to the EncoderLayer and 
# DecoderLayer classes. Passing them as arguments is more of a design choice than a necessity. Since 
# EncodeLayer is a common abstraction that can act on any kind of layers, it is reasonable to create encoder 
# as a container and pass the layers as arguments to the container. 
class EncoderLayer(nn.Module):
    def __init__(self, 
                 self_attention: MultiHeadedAttention, 
                 feed_forward: FeedForwardNN, 
                 d_model: int, 
                 dropout_prob: float):
        super().__init__()
        self.d_model = d_model
        self.dropout_prob = dropout_prob
        # These modules are now the child modules of the EncoderLayer and will be registered as parameters of the EncoderLayer.
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        # We need two instances of the SubLayerWrapper class. One for the self_attention and the other for the feed_forward.
        self.sublayer_wrappers = clone_module(module=SubLayerWrapper(d_model=self.d_model, dropout_prob=self.dropout_prob), num_clones=2)

    def forward(self, input: Tensor, mask: Tensor) -> Tensor:
        """This method is the forward pass of the EncoderLayer class.

        Args:
            input (Tensor): Source sequences provided as input to the EncoderLayer. These are the embeddings of the source 
                            sequences for the first EncoderLayer.
                            SHAPE: [batch_size, src_seq_len, d_model]
            mask (Tensor): Boolean mask to be applied to the input during attention scores calculation.
                           SHAPE: [batch_size, 1, 1, src_seq_len]
        Returns:
            Tensor: Output of the EncoderLayer.
                    SHAPE: [batch_size, src_seq_len, d_model]
        """
        # We are just saving the function call to the self_attention method in a variable and passing the
        # lambda function (contained within the variable) to the sublayer_wrappers[0] to execute it when 
        # needed.
        output = self.sublayer_wrappers[0](input, lambda input: self.self_attention(query_input=input, key_input=input, value_input=input, mask=mask))
        return self.sublayer_wrappers[1](output, self.feed_forward)
    

class Encoder(nn.Module):
    def __init__(self, encoder_layer: EncoderLayer, num_layers: int):
        super().__init__()
        self.encoder_layers = clone_module(module=encoder_layer, num_clones=num_layers)
        self.layer_norm = nn.LayerNorm(encoder_layer.d_model)

    def forward(self, input: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        """This method is the forward pass of the Encoder class. The output of the current EncoderLayer is
           passed as input to the next EncoderLayer. We have 6 identical EncoderLayers stacked on top of 
           each other. The output of the last EncoderLayer is passed through a Layer Normalization layer
           and returned as the final output of the Encoder

        Args:
            input (Tensor): Input to the Encoder i.e., embeddings of the tokenized src sequences.
                            input: [batch_size, src_seq_len, d_model]
            mask (Optional[Tensor], optional): Boolean mask to be applied during attention scores calculation.
                                               mask: [batch_size, 1, 1, src_seq_len]. Defaults to None.
                            
        Returns:
            Tensor: Output of the Encoder i.e., encoded src sequences.
                    output: [batch_size, src_seq_len, d_model]
        """
        output = input
        for idx, encoder_layer in enumerate(self.encoder_layers):
            logger.debug(f"POINT 1 -- Inside the forward pass of the Encoder. Passing input to EncoderLayer number {idx + 1}")
            # Pass the output of the previous EncoderLayer to the current EncoderLayer.
            output = encoder_layer(input=output, mask=mask)
        return self.layer_norm(output)