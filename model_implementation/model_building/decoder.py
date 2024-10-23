# This file implements the Decoder in the Transformer model. The Decoder is a stack of N=6 identical DecoderLayers.
# Refer to 'step_13_decoder.ipynb' (link to the notebook) notebook for a detailed explanation of each line of code 
# in this file.

from model_implementation.model_building.feed_forward_nn import FeedForwardNN
from model_implementation.model_building.multi_headed_attention import MultiHeadedAttention
from model_implementation.model_building.sublayer_wrapper import SubLayerWrapper
from model_implementation.utils.helpers import clone_module
from model_implementation.utils.logger import get_logger

from torch import nn, Tensor
from typing import Optional

logger = get_logger(__name__)

# (seq_len - 1) in the decoder input and mask comes from the fact that we remove the last token from the 
# decoder input when we create batches and masks. Refer to 'step_5_data_batching_and_masking.ipynb' notebook
# to understand this better.
#
# Here, lets try to understand how the shapes change when we calculate source_attention above.
# queries from tgt:    [batch_size, num_heads, seq_len - 1, d_k]
# keys from encoder:   [batch_size, num_heads, seq_len, d_k]
# values from encoder: [batch_size, num_heads, seq_len, d_k]
#
# attention_scores = queries * keys^{transpose}  --> * here represents matrix multiplication.
# attention_scores: [batch_size, num_heads, seq_len - 1, seq_len]
#
# attention_heads = attentions_scores * values  --> * here represents matrix multiplication.
# attention_heads: [batch_size, num_heads, seq_len - 1, d_k]
# output of source attention calculation: [batch_size, num_heads, seq_len - 1, d_k]
class DecoderLayer(nn.Module):
    def __init__(self, 
                 self_attention: MultiHeadedAttention, 
                 src_attention: MultiHeadedAttention, 
                 feed_forward: FeedForwardNN, 
                 d_model: int, 
                 dropout_prob: float):
        super().__init__()
        self.d_model = d_model
        self.dropout_prob = dropout_prob
        # These modules are now the child modules of the DecoderLayer and will be registered as parameters of the DecoderLayer.
        self.self_attention = self_attention
        self.src_attention = src_attention
        self.feed_forward = feed_forward
        self.sublayer_wrappers = clone_module(module=SubLayerWrapper(d_model=d_model, dropout_prob=dropout_prob), num_clones=3)

    def forward(self, input: Tensor, encoded_src: Tensor, tgt_mask: Tensor, src_mask: Optional[Tensor]=None) -> Tensor:
        """This method is the forward pass of the DecoderLayer class.

        Args:
            input (Tensor): Target sentence provided as input to the DecoderLayer. These are the embeddings of the target 
                            sentence for the first DecoderLayer.
                            SHAPE: [batch_size, seq_len - 1, d_model]
            encoded_src (Tensor): Encoded source sentence. This is the output of the Encoder. This is used to calculate the
                                  source attention scores for the target sentence. 
                                  SHAPE: [batch_size, seq_len, d_model] 
            tgt_mask (Tensor): Mask to prevent the future tokens in the target sentence to attend to the previous tokens and
                               also to prevent padding tokens from attending to any other token except other padding tokens.
                               SHAPE: [batch_size, 1, seq_len - 1, seq_len - 1]
            src_mask (Tensor, optional): Mask to prevent the the padding tokens to attend to the tokens in the tgt sentence. 
                                         Defaults to None.
                                         SHAPE: [batch_size, 1, seq_len, seq_len]

        Returns:
            Tensor: Returns the output of the DecoderLayer. This is the output of the Positionwise FeedForward Neural Network.
                    SHAPE: [batch_size, seq_len - 1, d_model]
        """
        # First sublayer: Self-Attention on the target sentence. Hence, it uses the tgt_mask.
        self_attention_output = self.sublayer_wrappers[0](input=input, sublayer=lambda input: self.self_attention(query_input=input, key_input=input, value_input=input, mask=tgt_mask)) 
        # To give intuition about src_attention, I have a query for a token in the target sentence. I want to know whether 
        # some token in the source sentence is important for me to predict the output for this token in the target sentence. 
        # So, I go to the source sentence and get the values for all the tokens in the source sentence. I then calculate 
        # the attention scores between the query (in tgt) and the keys (in src). I then calculate the attention heads for 
        # the token in the target sentence using the attention scores. This is what is done in the below line. Note that 
        # referring to statement 'the keys and values are from the source' doesn't mean that you get keys and values 
        # explicitly. It means we use the encoded data from the source sentence to calculate the queries and keys for 
        # this transformation.
        # Second sublayer: Attention on the source sentence. Hence, it uses the src_mask.
        src_attention_output = self.sublayer_wrappers[1](input=self_attention_output, sublayer=lambda self_attention_output: self.src_attention(query_input=self_attention_output, key_input=encoded_src, value_input=encoded_src, mask=src_mask))
        # Third sublayer: Positionwise FeedForward Neural Network.
        return self.sublayer_wrappers[2](input=src_attention_output, sublayer=self.feed_forward)
    


class Decoder(nn.Module):
    def __init__(self, decoder_layer: DecoderLayer, num_layers: int):
        super().__init__()
        self.decoder_layers = clone_module(module=decoder_layer, num_clones=num_layers)
        self.layer_norm = nn.LayerNorm(decoder_layer.d_model)

    def forward(self, input: Tensor, encoded_src: Tensor, tgt_mask: Tensor, src_mask: Optional[Tensor]=None) -> Tensor:
        """This method is the forward pass of the Decoder class. The output of the current DecoderLayer is
           passed as input to the next DecoderLayer. We have 6 identical DecoderLayers stacked on top of 
           each other. The output of the Encoder (last EncoderLayer) is also passed as input to the 
           first DecoderLayer. The output of the last DecoderLayer is passed through a Layer Normalization 
           layer and returned as the final output of the Decoder.

        Args:
            input (Tensor): Input to the Decoder i.e., embeddings of the tokenized tgt sequences.
                            SHAPE: [batch_size, seq_len - 1, d_model]
            encoded_src (Tensor): output of the encoder i.e., encoded src sequences.
                                  SHAPE: [batch_size, seq_len, d_model]
            tgt_mask (Tensor): Boolean mask to be applied during self attention scores calculation.
                               SHAPE: [batch_size, 1, seq_len - 1, seq_len - 1].
            src_mask (Tensor, optional): Boolean mask to be applied during src attention scores calculation.
                                         SHAPE: [batch_size, 1, seq_len, seq_len]. Defaults to None.

        Returns:
            Tensor: Output of the Decoder.
                    SHAPE: [batch_size, seq_len - 1, d_model]
        """
        output = input
        for idx, decoder_layer in enumerate(self.decoder_layers):
            logger.debug(f"POINT 1 -- Inside the forward pass of the Decoder. Passing input to DecoderLayer number {idx + 1}")
            # Pass the output of the previous DecoderLayer to the current DecoderLayer.
            output = decoder_layer(input=output, encoded_src=encoded_src, tgt_mask=tgt_mask, src_mask=src_mask)
        return self.layer_norm(output)