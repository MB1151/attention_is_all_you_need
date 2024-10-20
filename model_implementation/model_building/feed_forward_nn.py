# This file contains the implementation of the Feed Forward Neural Network layer to be used
# in the Transformer model. Refer to 'step_10_feed_forward_neural_network.ipynb' (link to the notebook)
# notebook for a detailed explanation of how this layer works.

from model_implementation.utils.constants import DROPOUT_PROB
from torch import nn, Tensor

class FeedForwardNN(nn.Module):
    def __init__(self, d_model: int, d_feed_forward: int, dropout_prob: float = DROPOUT_PROB):
        super().__init__()
        self.linear_layer_1 = nn.Linear(in_features=d_model, out_features=d_feed_forward)
        self.linear_layer_2 = nn.Linear(in_features=d_feed_forward, out_features=d_model)
        self.dropout_layer = nn.Dropout(p=dropout_prob)

    def forward(self, input: Tensor) -> Tensor:
        """Passes the input through the Feed Forward Neural Network and returns the output 
           of the neural network.

        Args:
            input (Tensor): The output of the Multi-Headed Attention layer.
                            shape: [batch_size, seq_len, d_model]

        Returns:
            Tensor: The output of the Feed Forward Neural Network.
                    shape: [batch_size, seq_len, d_model]
        """
        # We first expand the input to higher dimension. We apply the ReLU activation function in this layer.
        intermediate_output = self.linear_layer_1(input).relu()
        # Dropout layer to prevent overfitting
        intermediate_output = self.dropout_layer(intermediate_output)
        # We then compress the input back to its original dimension. There is no specific intuitive explanation 
        # as to why this is done. It is just shown to be working practically in neural networks in general and 
        # in this paper in particular.
        return self.linear_layer_2(intermediate_output)