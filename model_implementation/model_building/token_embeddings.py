# This file implements the embedding layer for the transformer model. The embedding layer is used to convert the token
# indices to their corresponding embedding vectors. The embedding vectors are then used as input to the transformer model.
#
# In the forward function, the embeddings are scaled by 'math.sqrt(embedding_dim)'. I have not found any solid reasoning
# as to why the embeddings are scaled by this factor. The discussion here (https://datascience.stackexchange.com/a/88159) 
# attempts to explain the reason for scaling the embeddings but that is incorrect.
#
# In general, the word_embeddings in the nn.Embedding layer are initialized using N(0, 1) distribution. You can find the 
# evidence for this in the source code of Embedding class in pytorch (https://pytorch.org/docs/stable/_modules/torch/nn/modules/sparse.html#Embedding) 
# Within the above source code, the 'weight' property (which are our word embeddings) is initalized inside the 
# 'reset_parameters' method using N(0, 1) distribution.
#  
# So, the expected magnitude of the embedding vector is sqrt(embedding_dim) and the expected magnitude of the positional 
# embedding is roughly (assuming uniform distribution for sinsuodial positional encodings which is not corrrect but gives 
# an easier estimate) sqrt(embedding_dim / 3). So, they are already on the same scale and embeddings don't need to be 
# scaled to bring them to the same scale. Use ChatGPT / Gemini (gave me a reasonable answer) to get an explanation on 
# how the expected magnitudes are calculated in the respective cases.
#
# This blog (https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec) just says that this 
# scaling is done to magnify the contribution of word embeddings when word embeddings are added to the positional 
# encodings. This explanation makes sense theoretically true, however, I have seen people mentioning on the internet 
# that scaling did not have any visible impact on their models (To be verified).
#
# Please refer to step_6_word_embeddings.ipynb (link to the notebook) notebook to understand details about each 
# step in the below code.
from torch import nn, Tensor

import math

class Embeddings(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        """Creates the embedding layer that serves as a look-up table for the tokens in the transformer model.

        Args:
            vocab_size (int): Size of the vocabulary i.e., number of distinct tokens in the vocabulary.
            embedding_dim (int): The size of the embedding vector to be generated for each token.
        """
        super(Embeddings, self).__init__()
        self.look_up_table = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    # The input is a '2D' tensor where each '1D' tensor within the '2D' tensor is the list
    # of indices corresponding to the tokens in the vocab.
    # [[0, 123, 3455, 4556, 7, 1, 2, 2], [0, 56, 98, 6234, 909, 56, 1, 2]]
    # 0 - <sos>, 1 - <eos>, 2 - <pad>
    def forward(self, input: Tensor) -> Tensor:
        """Returns the embedding vectors for the corresponding token indices in the input tensor.

        Args:
            input (Tensor): The input tensor containing token indices.
                            shape: [batch_size, seq_len]

        Returns:
            Tensor: The tensor of embedding vectors for the corresponding input token ids.
                    shape: [batch_size, seq_len, embedding_dim]
        """
        # There is no reasoning as to why the original 'attention_is_all_you_need' paper scaled the
        # embeddings using 'math.sqrt(embedding_dim)'. A few blogs attempted to explain this reasoning,
        # but I haven't found any correct explanation with solid reasoning.
        return self.look_up_table(input) * math.sqrt(self.embedding_dim)