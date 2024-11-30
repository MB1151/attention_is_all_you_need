# This file implements the machine translation transformer model that takes in tokenized src sentences,
# tokenized tgt sentences from the training set and outputs the predicted tgt translation sentence 
# tokens. Refer to 'step_15_machine_translation_model.ipynb' (add link to the notebook) to understand
# how MachineTranslationModel works.

from model_implementation.model_building.decoder import Decoder, DecoderLayer
from model_implementation.model_building.encoder import Encoder, EncoderLayer
from model_implementation.model_building.feed_forward_nn import FeedForwardNN
from model_implementation.model_building.multi_headed_attention import MultiHeadedAttention
from model_implementation.model_building.positional_encodings import PositionalEncoding
from model_implementation.model_building.token_embeddings import Embeddings
from model_implementation.model_building.token_predictor import TokenPredictor
from torch import nn, Tensor

import copy

class MachineTranslationModel(nn.Module):
    """Model that combines the Encoder, Decoder and the TokenPredictor to create a machine translation Transformer model."""

    def __init__(self, 
                 d_model: int, 
                 d_feed_forward: int, 
                 dropout_prob: float, 
                 num_heads: int, 
                 src_vocab_size: int, 
                 tgt_vocab_size: int, 
                 num_layers: int, 
                 max_seq_len: int):
        """Initializes the Transformer model.

        Args:
            d_model (int): size of the embedding vectors in the model.
            d_feed_forward (int): Number of neurons in the hidden layer of the feed forward neural network.
            dropout_prob (float): probability with which to drop data for regularization in the transformer model.
            num_heads (int): number of attention heads in each of the multi-head attention layers in the model.
            src_vocab_size (int): size of the source vocabulary.
            tgt_vocab_size (int): size of the target vocabulary.
            num_layers (int): number of layers in the Encoder and Decoder.
            max_seq_len (int): Maximum length of the sequence that is ever input to the model.
        """
        super(MachineTranslationModel, self).__init__()
        self.src_embedding = Embeddings(vocab_size=src_vocab_size, embedding_dim=d_model)
        self.tgt_embedding = Embeddings(vocab_size=tgt_vocab_size, embedding_dim=d_model)
        # We have to create two instances of the PositionalEncoding since PositionalEncoding module has a Dropout layer
        # and is applied independently in both the cases.
        self.src_positional_encoding = PositionalEncoding(encoding_size=d_model, dropout_prob=dropout_prob, max_len=max_seq_len)
        self.tgt_positional_encoding = PositionalEncoding(encoding_size=d_model, dropout_prob=dropout_prob, max_len=max_seq_len)
        # Note that multi_headed_attention, feed_forward_nn, encoder_layer and decoder_layer are not child modules of
        # the MachineTranslationModel class. They are just variables that are used to create the child modules of the
        # MachineTranslationModel class.
        multi_headed_attention = MultiHeadedAttention(num_heads=num_heads, d_model=d_model, dropout_prob=dropout_prob)
        feed_forward_nn = FeedForwardNN(d_model=d_model, d_feed_forward=d_feed_forward, dropout_prob=dropout_prob)
        encoder_layer = EncoderLayer(self_attention=copy.deepcopy(multi_headed_attention), 
                                     feed_forward=copy.deepcopy(feed_forward_nn), 
                                     d_model=d_model, 
                                     dropout_prob=dropout_prob)
        decoder_layer = DecoderLayer(self_attention=copy.deepcopy(multi_headed_attention), 
                                     src_attention=copy.deepcopy(multi_headed_attention),
                                     feed_forward=copy.deepcopy(feed_forward_nn), 
                                     d_model=d_model, 
                                     dropout_prob=dropout_prob)
        # encoder, decoder and token_predictor are the child modules of the MachineTranslationModel class.
        self.encoder = Encoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.decoder = Decoder(decoder_layer=decoder_layer, num_layers=num_layers)
        self.token_predictor = TokenPredictor(d_model=d_model, tgt_vocab_size=tgt_vocab_size)
        self.initialize_model_parameters()

    def initialize_model_parameters(self):
        """Initializes the parameters of the model using the Xavier Uniform initialization."""
        for params in self.parameters():
            # This is to ensure the only the weights are initialized and not the biases. biases usually have only
            # one dimension and the weights have more than one dimension. biases are usually initialized to zero.
            if params.dim() > 1:
                nn.init.xavier_uniform_(params)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        """The forward pass of the Transformer model. The source sentences are passed through the Encoder and the target
           sentences are passed through the Decoder. The output of the Decoder is passed through the token predictor to
           get the probability distribution over the target vocabulary.

        Args:
            src (Tensor): Source sequences (English) containing the token ids corresponding to the indices in the src vocabulary. 
                          Example input looks like [[0, 4, 55, 67, 1, 2, 2], [0, 42, 585, 967, 19, 26, 1]]
                          SHAPE: [batch_size, src_seq_len]
            tgt (Tensor): Target sequences (Telugu) containing the token ids corresponding to the indices in the tgt vocabulary. 
                          Example input looks like [[0, 3, 5, 677, 81, 1, 2], [0, 7, 67, 190, 3245, 1]]
                          SHAPE: [batch_size, tgt_seq_len]
            src_mask (Tensor): Mask to be applied to the source sequences in each of the attention heads.
                               src_mask: [batch_size, 1, src_seq_len, src_seq_len]
            tgt_mask (Tensor): Mask to be applied to the target sequences in each of the attention heads.
                               tgt_mask: [batch_size, 1, tgt_seq_len - 1, tgt_seq_len - 1]

        Returns:
            Tensor: Log probability distribution over the tokens in the target vocabulary (Telugu vocabulary).
                    SHAPE: [batch_size, tgt_seq_len - 1, tgt_vocab_size]
        """
        # Pass the source sentences through the encoder to get the encoded source token vectors.
        encoded_src = self.encode(src=src, src_mask=src_mask)
        # Pass the target sentence through the decoder to get the encoded target token vectors.
        decoded_tgt = self.decode(tgt=tgt, tgt_mask=tgt_mask, encoded_src=encoded_src, src_mask=src_mask)
        return self.generate_tgt_token_prob_distributions(decoded_tgt=decoded_tgt)

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """Encodes the source sentences (English).

        Args:
            src (Tensor): A batch of source sequences containing the token ids corresponding to the indices in the src vocabulary.
                          SHAPE: [batch_size, src_seq_len]
            src_mask (Tensor): Mask to be applied to the source sequences in each of the attention heads. Same mask will be 
                               applied to the sequence in all the attention heads.
                               SHAPE: [batch_size, 1, 1, src_seq_len]

        Returns:
            Tensor: Encoded source sequences. Each token in the source sequence is represented by a vector that encodes
                    all the information about the token and its relationship with other tokens in the sequence.
                    SHAPE: [batch_size, src_seq_len, d_model]
        """
        # Get the embeddings for the source sentences.
        src_embeddings = self.src_embedding(src)
        print(f"shape of src_embeddings: {src_embeddings.shape}")
        print(f"src_embeddings: {src_embeddings}")
        print("-" * 150)
        # Add the positional encodings to the embeddings.
        src_embeddings = self.src_positional_encoding(src_embeddings)
        print(f"shape of src_embeddings after positional encoding: {src_embeddings.shape}")
        print(f"src_embeddings after positional encoding: {src_embeddings}")
        print("-" * 150)
        # Pass the source sentence through the encoder.
        encoded_src = self.encoder(input=src_embeddings, mask=src_mask)
        return encoded_src

    def decode(self, tgt: Tensor, encoded_src: Tensor, src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        """Encodes each token in the target sequence using the information from the source sequences and the 
        assocations between tokens in the target sequences.

        Args:
            tgt (Tensor): A batch of target sequences containing the token ids corresponding to the indices in the tgt vocabulary.
                          SHAPE: [batch_size, tgt_seq_len]
            encoded_src (Tensor): The encoded token representations of the source sequences. This is used to calculate the
                                  source attention scores for the target sentence.
                                  SHAPE: [batch_size, src_seq_len, d_model]
            src_mask (Tensor): Mask to be applied to the source sequences in each of the attention heads. Same mask will be 
                               applied to the sequence in all the attention heads.
                               SHAPE: [batch_size, 1, src_seq_len, src_seq_len]
            tgt_mask (Tensor): Mask to be applied to the target sequences in each of the attention heads. Same mask will be 
                                         applied to the sequence in all the attention heads.
                                         SHAPE: [batch_size, 1, tgt_seq_len - 1, tgt_seq_len - 1]
                               
        Returns:
            Tensor: Encoded (or Decoded if that makes more sense to you) target sequences. Each token in the target 
                    sequence is represented by a vector that encodes all the information about the token and its 
                    relationship with other tokens in the target sequence and the corresponding source sequences.
                    SHAPE: [batch_size, tgt_seq_len - 1, d_model]
        """
        # Get the embeddings for the target sequences.
        tgt_embeddings = self.tgt_embedding(tgt)
        # Add the positional encodings to the embeddings.
        tgt_embeddings = self.tgt_positional_encoding(tgt_embeddings)
        # Pass the target sequence through the decoder.
        decoded_tgt = self.decoder(input=tgt_embeddings, encoded_src=encoded_src, tgt_mask=tgt_mask, src_mask=src_mask)
        return decoded_tgt

    def generate_tgt_token_prob_distributions(self, decoded_tgt: Tensor) -> Tensor:
        """Takes the output of the decoder and generates the probability distribution for each token over the target vocabulary.

        Args:
            decoded_tgt (Tensor): The output of the decoder. Each token in the target sequence is represented by a vector that
                                  encodes all the information about the token and its relationship with other tokens in the 
                                  target sequence and the corresponding source sequences.
                                  SHAPE: [batch_size, tgt_seq_len - 1, d_model]

        Returns:
            Tensor: Log probability distribution over the tokens in the target vocabulary (Telugu vocabulary in this case).
        """
        # Convert the output of the decoder to the probability distribution over the target vocabulary. This will be
        # used to calculate the loss in the training phase.
        return self.token_predictor(decoded_tgt)