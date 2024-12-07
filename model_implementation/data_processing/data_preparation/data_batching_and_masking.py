# This file implements the data batching and masking. It takes in the input data and creates the data
# batches and masks to be used with the model.

from model_implementation.utils.logger import get_logger
from torch import Tensor

import torch

logger = get_logger(__name__)

def construct_padding_mask(input: Tensor, pad_token_id: int) -> Tensor:
    """Creates a mask to prevent the padding tokens from attending to the actual tokens in the input.

    Args:
        input (Tensor): A tensor where each row corresponds to a sentence. Contains token ids.
                        shape: [batch_size, seq_len].
        pad_token_id (int): Id of the padding token. Usually 2.

    Returns:
        Tensor (bool): padding mask for the given input tensor.
                       shape: [batch_size, 1, seq_len].
    """
    mask = (input != pad_token_id)
    mask = mask.unsqueeze(1)
    return mask


def construct_look_ahead_mask(size: int) -> Tensor:
    """Creates a mask to prevent the tokens appearing after the current token to attend to the current 
       token or any token before it.

    Args:
        size (int): Size of the mask to be created which is generally the length of the sentence or
                    the number of tokens in the sentence.

    Returns:
        Tensor: look ahead mask for the given size.
                shape: [size, size].
    """
    attention_mask = torch.triu(torch.ones(size, size, dtype=torch.uint8), diagonal=1)
    return attention_mask == 0


class Batch:
    """Object for holding a batch of data and the corresponding mask to be used for training."""

    def __init__(self, src_batch: Tensor, tgt_batch: Tensor, pad_token_id: int, device: str):
        """Initialize the Batch object. Updates the tgt_batch to the format expected by the decoder
           during training. Also, creates the mask for the source and target sentences.

        Args:
            src_batch (Tensor): Tensor containing the source sentences in the batch. Contains token ids.
                                shape: [batch_size, src_seq_len].
            tgt_batch (Tensor): Tensor containing the target sentences in the batch. Contains token ids.
                                shape: [batch_size, tgt_seq_len].
            pad_token_id (int): Id of the pad token appended to the sentences in the batch. Usually 
                                set to 2.
            device (str): Device to be used for storing the tensors. Can be either 'cpu' or 'cuda'.
        """
        # Store the device to be used for storing the tensors.
        self.device = device
        self.src = src_batch.to(device)
        # It might be tempting to make the src_mask of shape [batch_size, 1, src_seq_len, src_seq_len] but, for the 
        # src_mask to be used with self attention in Encoder, the shape should be 
        # [batch_size, num_heads, src_seq_len, src_seq_len] and for the src_mask to be used with source attention in
        # Decoder, the shape should be [batch_size, num_heads, tgt_seq_len, src_seq_len]. So, we will keep the 
        # shape of src_mask as [batch_size, 1, 1, src_seq_len] and then let the model handle the broadcasting of 
        # the mask to the required shape.
        # The source sequences only need the padding mask since the Encoder does not have to predict
        # the next token in the sentence but just encode the input to be used by the Decoder.
        # Shape of src_mask: [batch_size, 1, 1, src_seq_len]
        self.src_mask = construct_padding_mask(input=src_batch, pad_token_id=pad_token_id).unsqueeze(1).to(device)
        # Removes the last token (<eos> or <pad>) from the target sequences to create the target_decoder_input.
        # Shape of tgt_decoder_input: [batch_size, tgt_seq_len]
        self.tgt_decoder_input = tgt_batch[:, :-1].to(device)
        # Removes the first token (<sos>) from the target sequences to create the target_expected_decoder_output.
        # Shape of tgt_expected_decoder_output: [batch_size, tgt_seq_len]
        self.tgt_expected_decoder_output = tgt_batch[:, 1:].to(device)
        # Shape of tgt_mask: [batch_size, 1, tgt_seq_len, tgt_seq_len]
        self.tgt_mask = self.construct_target_mask(tgt=self.tgt_decoder_input, pad_token_id=pad_token_id).unsqueeze(1).to(device)
        # Number of tokens in the target sequences excluding the padding tokens. This is used during model 
        # training for the loss calculation inorder to normalize the total loss and find the loss per token.
        self.non_pad_tokens = (self.tgt_expected_decoder_output != pad_token_id).sum()


    def construct_target_mask(self, tgt: Tensor, pad_token_id: int) -> Tensor:
        """Creates the mask for the target sequences. There is really no value in using the padding mask in 
           the target sequences. The padding tokens are only at the end of the sequence and the future 
           tokens are not allowed to attend to the earlier tokens because of the look_ahead_mask already. 
           This means the padding tokens are already masked for the actual tokens (part of the sequence) 
           because of the look_ahead_mask. In other words, the padding_mask is not changing anything in the 
           masks for the rows where the tokens are non-padding tokens. The padding_mask only changes the 
           target_mask for the rows where the tokens are padding tokens. This anyway doesn't matter 
           because we don't really use the output of the padding tokens in the final loss calculation.

           However, I don't know why most of the implementations still go ahead and use padding mask in the
           target sequences. I am not sure if I am missing something here. So, I am just leaving the code
           as it is incase it is important. If you know the reason, please let me know.

        Args:
            tgt (Tensor): Tensor containing the tgt sequences in the batch. Contains token ids.
                          shape: [batch_size, tgt_seq_len - 1].
            pad_token_id (int): Id of the pad token appended to the sequences in the batch. Usually 
                                set to 2.

        Returns:
            Tensor: target mask for the given target tensor.
                    shape: [batch_size, tgt_seq_len - 1, tgt_seq_len - 1].
        """
        # The target sentences need both the padding mask and the look ahead mask. The padding mask is used
        # to prevent the padding tokens from attending to the other tokens in the target sentences. The look
        # ahead mask is used to prevent the future tokens from attending to the current token or any token.
        # Shape of tgt_mask after this step: [batch_size, tgt_seq_len - 1, tgt_seq_len - 1]
        tgt_mask = construct_padding_mask(input=tgt, pad_token_id=pad_token_id).repeat(1, tgt.size(1), 1)
        tgt_mask = tgt_mask & construct_look_ahead_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask