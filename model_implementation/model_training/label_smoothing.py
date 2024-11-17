# This file implements Label Smoothing which is a regularization technique used to prevent the model from
# overfitting to the training data. Refer to 'step_16_label_smoothing.ipynb' (add link to the notebook) 
# for a detailed explanation of each line of code in this file.

from model_implementation.utils.constants import DEVICE_CPU, SMOOTHING_PROB
from model_implementation.utils.logger import get_logger
from torch import nn, Tensor

import torch

logger = get_logger(__name__)

# Combining the above steps into a module to be used in the transformer implementation.
class LabelSmoothing(nn.Module):
    def __init__(self, tgt_vocab_size: int, padding_idx: int, smoothing: float=SMOOTHING_PROB, device: str=DEVICE_CPU):
        super(LabelSmoothing, self).__init__()
        # Number of classes in the classification problem. It is the size of the vocabulary in transformers.
        self.vocab_size = tgt_vocab_size
        # Index of the padding token or the class label for the padding token. Usually set to 2.
        self.padding_idx = padding_idx
        # Amount of probability to be shared among the tokens excluding correct token and padding tokens.
        self.smoothing = smoothing
        # Amount of probability shared with the correct token.
        self.confidence = 1 - smoothing
        # Device to be used for storing the tensors.
        self.device = device
    
    def forward(self, targets: Tensor) -> Tensor:
        """Calculates the smoothed probabilities for each of the target tokens within each sentence.

        Args:
            targets (Tensor): The target tensor containing the correct class labels (expected token indices from the 
                              vocab) for each token in the batch. An example target tensor for a batch of 2 sentences
                              each with 8 tokens and 6 possible classes for prediction (including the padding token)
                              would be: [[0, 3, 4, 5, 5, 1, 2, 2], [1, 5, 3, 3, 4, 0, 0, 2]]
                              SHAPE: [batch_size, tgt_seq_len - 1]

        Returns:
            Tensor: A smoothed probability distribution (1D tensor) for each target token in the batch.
                    SHAPE: [batch_size, tgt_seq_len - 1, vocab_size]                    
        """
        logger.debug(f"POINT 0 -- device: {self.device}")
        # The above description showing the shape as (tgt_seq_len - 1) is because the first token is removed from the
        # target tensor while calculating the loss. 'tgt_seq_len' variable here is the number of tokens in each 
        # target sequence in the batch before we removed the first token to form the expected decoder output. 
        # Don't get confused with the variable naming. Just ignore this explanation if it is confusing.
        batch_size, tgt_seq_len = targets.shape
        # Creating a tensor that will hold the smoothed probabilities for each target token in all the sentences.
        smoothed_probs = torch.zeros(size=(batch_size, tgt_seq_len, self.vocab_size), dtype=torch.float32, device=self.device)
        logger.debug(f"POINT 1 -- smoothed_probs device: {smoothed_probs.device}")
        # Filling the entire tensor with the smoothing probability. We will deal with the probabilities of the
        # correct token and padding token later. We use 'vocab_size - 2' because we don't want to assign the
        # smoothing probability to the correct token and the padding token.
        smoothed_probs = smoothed_probs.fill_(value=self.smoothing / (self.vocab_size - 2))
        # Bringing the targets tensor to contain the same number of dimensions as the smoothed_probs tensor to 
        # use it with the 'scatter_' function. This is to replace the probabilities in the smoothed_probs tensor 
        # for the padding token and the correct token in the following steps.
        unsqueezed_targets = targets.unsqueeze(dim=-1)
        logger.debug(f"POINT 2 -- unsqueezed_targets device: {unsqueezed_targets.device}")
        # Replacing the probabilities in the smoothed_probs tensor with the confidence probability at the 
        # positions that correspond to the correct class labels (expected output tokens in the target).
        smoothed_probs.scatter_(dim=-1, index=unsqueezed_targets, value=self.confidence)
        # The padding token should not be predicted at all by the model. So, the probability associated with the
        # class label that correspond to the padding token within each target token distribution should be 0. 
        smoothed_probs[:, :, self.padding_idx] = 0
        # The target tensor is appended with the padding tokens at the end. These are just dummy tokens added to bring 
        # all the sentences in the batch to the same length. We don't want the model to consider these tokens at all 
        # in the loss calculation. So, we set the probabilities of the entire rows corresponding to the padding tokens
        # to 0. More about why this setup works is explained in the notebook 'step_17_loss_computation.ipynb'.
        mask = unsqueezed_targets.repeat(1, 1, self.vocab_size) == self.padding_idx
        return smoothed_probs.masked_fill(mask=mask, value=0.0)