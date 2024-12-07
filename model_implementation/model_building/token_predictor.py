# This file implements token predictor which converts the decoder output into probabilities over
# the target vocabulary. Refer to 'step_14_token_predictor.ipynb' (link to the notebook) to 
# understand hwo this class works.

from torch import nn, Tensor

class TokenPredictor(nn.Module):
    def __init__(self, d_model: int, tgt_vocab_size: int):
        super(TokenPredictor, self).__init__()
        self.d_model = d_model
        self.vocab_size = tgt_vocab_size
        self.linear = nn.Linear(in_features=d_model, out_features=tgt_vocab_size)
        # The non-module variables are not added to the list of parameters of the model.
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, decoder_output: Tensor) -> Tensor:
        """The forward pass of the token predictor. Calculates the probability distribution over the 
           vocabulary. Each token vector has a corresponding probability distribution over the 
           vocabulary since we predict one token per output.

        Args:
            decoder_output (Tensor): Output of the Decoder.
                                     SHAPE: [batch_size, tgt_seq_len, d_model]

        Returns:
            Tensor: Log probability distribution over the vocabulary. 
                    SHAPE: [batch_size, tgt_seq_len, vocab_size]
        """
        # Project the decoder output to the vocab_size dimensional space.
        logits = self.linear(decoder_output)
        # Convert the logits to a probability distribution over the vocabulary. All the entries in the
        # output tensor are negative since we are using log softmax. The log softmax is used to make
        # the training more numerically stable. However, the maximum value in log_softmax is still the 
        # same as the maximum value of the general softmax output.
        return self.log_softmax(logits)