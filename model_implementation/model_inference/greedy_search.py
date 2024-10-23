
from model_implementation.model_building.machine_translation_model import MachineTranslationModel
from model_implementation.model_inference.base_search import SequenceSearchBase, SequenceState
from model_implementation.utils.logger import get_logger
from torch import Tensor
from typing import List

import torch


logger = get_logger(name=__name__)


class GreedySearch(SequenceSearchBase):
    def __init__(self, translation_model: MachineTranslationModel, tgt_seq_limit: int, sos_token_id: int, eos_token_id: int, device: str):
        super().__init__(translation_model=translation_model, 
                         tgt_seq_limit=tgt_seq_limit, 
                         sos_token_id=sos_token_id, 
                         eos_token_id=eos_token_id, 
                         device=device)


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
        # Move the source batch and mask to the device. This is done in an effort move the tensors to GPU.
        # If the device is 'cpu', then the tensors will be stored on CPU.
        src_batch = src_batch.to(self.device)
        src_mask = src_mask.to(self.device)
        batch_size = src_batch.size(0)
        # Pass the source sentence through the encoder to find the encoded src sentence tokens.
        encoded_src = self.translation_model.encode(src=src_batch, src_mask=src_mask)
        self.initialize_search_state(batch_size=batch_size)
        # We generate 'tgt_seq_limit' number of tokens (the maximum allowed) in the target sequences.
        for _ in range(self.tgt_seq_limit):
            # If there are no potential tgt sequences, we stop the greedy search. This essentially means
            # we already found <eos> token in all the tgt sequences.
            if len(self.search_state.running_state) == 0:
                break
            # As we proceed forward in the algorithm, some of the src sequences might have already found have found <eos> 
            # token in its corresponding tgt sequence and so these should not be used anymore for further predictions. 
            src_for_inference, src_mask_for_inference = self.get_src_for_running_state(encoded_src=encoded_src, src_mask=src_mask)
            # Gets the corresponding tgt sequences and theirs masks as a tensor to be used with the
            # translation model. 
            tgt_for_inference, tgt_mask_for_inference = self.get_tgt_for_running_state()
            # Predict 1 token for all the running tgt sequences and update the search state.
            self.advance(encoded_src=src_for_inference, src_mask=src_mask_for_inference, tgt_batch=tgt_for_inference, tgt_mask=tgt_mask_for_inference)
        self.add_running_state_to_complete_state()
        return [self.search_state.complete_state[src_seq_idx].tokens[1:-1] for src_seq_idx in sorted(self.search_state.complete_state.keys())]


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
        predicted_log_probabilities = predicted_log_probabilities[:, -1, :]
        _, predicted_tokens = predicted_log_probabilities.max(dim=1, keepdim=True)
        new_running_state: List[SequenceState] = []
        for pred_token, seq_state in zip(predicted_tokens, self.search_state.running_state):
            seq_state.tokens = torch.cat((seq_state.tokens, pred_token), dim=0)
            if pred_token.item() != self.eos_token_id:
                new_running_state.append(seq_state)
            else:
                self.search_state.complete_state[seq_state.index] = seq_state
        self.search_state.running_state = new_running_state
        logger.debug(f"search state after update: {repr(self.search_state)}")


    def add_running_state_to_complete_state(self):
        """Adds the running sequences to complete state once the maximum allowed tokens are predicted and <eos> token
        is not predicted for any target sequence."""
        for seq_state in self.search_state.running_state:
            self.search_state.complete_state[seq_state.index] = seq_state