# This file implements the LossCompute class that is used to compute the KL Divergence loss for the 
# model predictions and the target tensor. Refer to 'step_17_loss_computation.ipynb' (link to the notebook)
# for a detailed explanation of each line of code in this file.

from torch import nn, Tensor

class LossCompute:
    def __init__(self):
        # We use the 'sum' reduction to sum the KL Divergence over all the tokens in all the sentences in the batch. 
        # The loss is then averaged over all the tokens in the batch to find the loss per token which is used as the 
        # objective function.         
        self.kl_div_loss = nn.KLDivLoss(reduction="sum")

    # The '__call__' method allows an object of the class to be called just like a function.
    def __call__(self, log_predictions: Tensor, targets: Tensor, num_non_pad_tokens: int) -> Tensor:
        """Computes the KL Divergence loss for the model predictions and the target tensor.

        Args:
            log_predictions (Tensor): The log of the model predictions for the target tokens in the batch.
                                      Each token has a probability distribution over the vocabulary.
                                      shape: [batch_size, seq_len, vocab_size]
            targets (Tensor): The expected target for the model predictions. The target tensor is a smoothed
                              probability distribution over the vocabulary for each token in the batch. 
                              shape: [batch_size, seq_len, vocab_size]
            num_non_pad_tokens (int): The number of non-pad tokens in the target of the batch.

        Returns:
            Tensor: The KL Divergence per token in the batch which is used as the objective function for model
                    training.
        """
        # Calculates the KL Divergence loss between the model predictions and the targets.
        kl_div_loss = self.kl_div_loss(input=log_predictions, target=targets)
        # Calculate the KL Divergence loss per token in the batch.
        return kl_div_loss / num_non_pad_tokens