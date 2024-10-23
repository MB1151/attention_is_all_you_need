
from abc import ABC, abstractmethod
from dataclasses import dataclass
from model_implementation.data_processing.data_preparation.data_batching_and_masking import construct_look_ahead_mask
from model_implementation.model_building.machine_translation_model import MachineTranslationModel
from model_implementation.utils.logger import get_logger
from torch import Tensor
from typing import Dict, List, Tuple

import torch

logger = get_logger(__name__)


@dataclass
class SequenceState:
    # Index of the source sentence for which the current target sequence has been predicted.
    index: int
    # Sequence of tokens in the target prediction.
    tokens: Tensor
    # Log of the probability that this is the current translation for the source sequence.
    log_prob: float

class SearchState:
    def __init__(self):
        # Holds the complete predictions for the source sequences keyed by the source sequence index in the batch.
        self.complete_state: Dict[int, SequenceState] = {}
        # Holds the active sequences that are active.
        self.running_state: List[SequenceState] = []

    def __repr__(self) -> str:
        return f"Complete State: {self.complete_state}\nRunning State: {self.running_state}"


class SequenceSearchBase(ABC):
    """A base class for the sequence search algorithms. All sequence search algorithms must inherit from this class."""
    def __init__(self, translation_model: MachineTranslationModel, tgt_seq_limit: int, sos_token_id: int, eos_token_id: int, device: str):
        """Initialized the SequenceSearch object.

        Args:
            translation_model (MachineTranslationModel): The translation model that is used for predictions.
            tgt_seq_limit (int): Maximum number of tokens that can be predicted for any target sequence. 
            sos_token_id (int): Token id for the start of the sentence token.
            eos_token_id (int): Token id for the end of the sentence token.
            device (str): Device to be used for storing the tensors. Can be either 'cpu' or 'cuda'.
        """
        self.translation_model = translation_model
        self.tgt_seq_limit = tgt_seq_limit
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.device = device


    @abstractmethod
    def decode(self, src_batch: Tensor, src_mask: Tensor) -> List[Tensor]:
        """Translates the source sentences to target sentences using sequence search algorithm.

        Args:
            src_batch (Tensor): A batch of source sentences to be translated.
                                SHAPE: [batch_size, seq_len]
            src_mask (Tensor): Mask to prevent the padding tokens in the source sentences to attend 
                               to the actual tokens in the sources sentences.
                               SHAPE: [batch_size, 1, seq_len, seq_len]
        Returns:
            List[Tensor]: Translated target sentences (contains token ids in the target vocabulary).
                          Length of list: batch_size
                          Shape of the tensors within the list: 1D tensors with different (likely) lengths. 
        """
        raise NotImplementedError


    @abstractmethod
    def update_search_state(self, predicted_log_probabilities: Tensor):
        """Updates the state of the search by appending the newly predicted tokens to the running tgt sequences. 
        If a tgt sequence is complete (<eos> token is predicted), it is removed from the running state and used to 
        update the complete state. If the tgt sequence is not complete, it is added back to the running state for 
        the next iteration (to predict one more token).

        Args:
            predicted_log_probabilities (Tensor): The probability distributions over the tgt vocabulary for 
                                                  each of the tokens in each of the current tgt sequences.
                                                  SHAPE: [BATCH_SIZE (varies), SEQUENCE_LENGTH (varies), TGT_VOCAB_SIZE]
        """
        raise NotImplementedError


    @abstractmethod
    def add_running_state_to_complete_state(self):
        """Adds the running sequences to complete state once the maximum allowed tokens are predicted and <eos> token
        is not predicted for any target sequence."""
        raise NotImplementedError
    

    def initialize_search_state(self, batch_size: int):
        """Initializes the search state and specifically the running state for each of src sequences.

        Args:
            batch_size (int): Number of src sequences to be translated to tgt language.
        """
        self.search_state = SearchState()
        self.search_state.running_state = [SequenceState(index=idx, tokens=torch.tensor(data=[self.sos_token_id], dtype=torch.int32, device=self.device), log_prob=0.0) for idx in range(batch_size)]


    def get_src_for_running_state(self, encoded_src: Tensor, src_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Creates a new encoded src tensor in which the src sequences match the indices in the running 
        state maintained by search algorithm. Also, creates the src masks for the corresponding src sequences.

        Args:
            encoded_src (Tensor): Encoded source sequences i.e., the output of the Encoder after passing the src_batch.
                                  SHAPE: [batch_size, seq_len, d_model]
            src_mask (Tensor): Mask for the corresponding encoded_src.
                               SHAPE: [batch_size, 1, seq_len, seq_len]
        Returns:
            Tuple[Tensor, Tensor]: src sequence batch tensor and the corresponding mask to be used with Decoder.
                                   SHAPE: ([len(running_state), seq_len, d_model], [len(running_state), 1, seq_len, seq_len])
        """
        # Finds the indices of all the src_sequences for which there is a running tgt sequence. If the same
        # src sequence has multiple tgt sequences, the same src sequence index appears multiple times.
        # Example src_indices: [0, 0, 3, 3, 3, 7, 7, 8]
        # The above src_indices means 
        # There are 2 running tgt sequences for the 0th src sequence.
        # There are 3 running tgt sequences for the 3rd src sequence.
        # There are 2 running tgt sequences for the 7th src sequence.
        # There is 1 running tgt sequence for the 8th src sequence.
        # The index is the index of the src_sequence in the 'src_batch' or the 'encoded_src'.
        src_indices = torch.tensor(data=[state.index for state in self.search_state.running_state], dtype=torch.int32, device=self.device)
        # Select the tensors corresponding to the indices in 'src_indices' and create a new
        # tensor to be used as a src for inference.
        # Example src_for_inference: [0th tensor, 0th tensor, 3rd tensor, 3rd tensor, 3rd tensor, 7th tensor, 7th tensor, 7th tensor, 8th tensor] 
        src_for_inference = torch.index_select(input=encoded_src, dim=0, index=src_indices).to(self.device)
        # Create the src mask similarly.
        src_mask_for_inference = torch.index_select(input=src_mask, dim=0, index=src_indices).to(self.device)
        return src_for_inference, src_mask_for_inference


    def create_tgt_mask_for_inference(self, batch_size: int, seq_len: int) -> Tensor:
        """Creates the target mask for inference. During inference, there are no padding tokens in the target sentence. So,
        the target mask is created without taking the padding tokens into account.

        Args:
            batch_size (int): Number of sequences in the batch.
            seq_len (int): Length of each sequence in the batch.

        Returns:
            Tensor: Target mask for inference. 
                    SHAPE: [batch_size, 1, seq_len, seq_len].
        """
        # Create a 2D look ahead mask for a sequence of length 'seq_len'.
        # The mask is of shape [seq_len, seq_len].
        tgt_mask = construct_look_ahead_mask(size=seq_len)
        # The same look ahead mask is applied to all the sequences in the batch. So, we unsqueeze the mask and repeat it
        # 'batch_size' times.
        # The mask is of shape [batch_size, seq_len, seq_len].
        tgt_mask = tgt_mask.unsqueeze(0).repeat(batch_size, 1, 1)
        # The same mask is applied to all the heads in the decoder. So, we unsqueeze the mask to make it broadcastable
        # across the heads.
        # The mask is of shape [batch_size, 1, seq_len, seq_len].
        tgt_mask = tgt_mask.unsqueeze(1)
        # The target decoder input does not contain the padding tokens. So, do not need to take padding into account to
        # create the target mask.
        return tgt_mask


    def get_tgt_for_running_state(self) -> Tuple[Tensor, Tensor]:
        """Creates a new tgt tensor by stacking all the tgt sequences in the running state maintained 
        by search algorithm. Also, creates the tgt masks for the corresponding tgt sequences.

        Returns:
            Tuple[Tensor, Tensor]: tgt sequence batch tensor and the corresponding mask to be used with Decoder.
                                   SHAPE: ([BATCH_SIZE (varies), SEQ_LEN (varies)], [BATCH_SIZE (varies), 1, SEQ_LEN (varies), SEQ_LEN (varies)])
        """
        tgt_for_inference = torch.stack(tensors=[state.tokens for state in self.search_state.running_state], dim=0).to(self.device)
        tgt_mask_for_inference = self.create_tgt_mask_for_inference(batch_size=tgt_for_inference.size(0), seq_len=tgt_for_inference.size(1)).to(self.device)
        logger.debug(f"tgt for inference: {tgt_for_inference}")
        logger.debug(f"tgt_mask for inference: {tgt_mask_for_inference}")
        return tgt_for_inference, tgt_mask_for_inference


    def advance(self, encoded_src: Tensor, src_mask: Tensor, tgt_batch: Tensor, tgt_mask: Tensor):
        """Predicts 1 token for all the (src, tgt) pairs and appends it at the back of each
        tgt sequence. Also, updates the search state according the newly predicted tokens and their
        associated probabilities.

        Args:
            encoded_src (Tensor): Encoded source sequences i.e., the output of the Encoder after passing the src_batch.
                                  SHAPE: [BATCH_SIZE (varies), SEQ_LEN (varies), d_model]
            src_mask (Tensor): Mask for the corresponding encoded_src.
                               SHAPE: [BATCH_SIZE (varies), 1, SEQ_LEN (varies), SEQ_LEN (varies)]
            tgt_batch (Tensor): The input tgt sequences for which a new token needs to be predicted.
                                SHAPE: [BATCH_SIZE (varies), SEQ_LEN (varies), d_model]
            tgt_mask (Tensor): Mask for the corresponding tgt_batch.
                               SHAPE: [BATCH_SIZE (varies), 1, SEQ_LEN (varies), SEQ_LEN (varies)]
        """
        # Run the decoder on the src and the corresponding running tgt sequences.
        decoded_tgt = self.translation_model.decode(tgt=tgt_batch, tgt_mask=tgt_mask, encoded_src=encoded_src, src_mask=src_mask)
        # Convert the decoder output into token probabilties.
        predicted_log_probabilities = self.translation_model.token_predictor(decoded_tgt)
        # Update the beam state according the predicted tokens for each of the tgt sequences.
        self.update_search_state(predicted_log_probabilities=predicted_log_probabilities)