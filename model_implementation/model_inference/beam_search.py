# This file implements BeamSearch for the given translation model. The concept of BeamSearch is relatively 
# simple but the implementation is tricky and might be little confusing. In addition, this might not be
# the best of Beam Search implementations. I have done my best to make it understandable by adding comments,
# but it might still be confusing to the reader. Please spend some time thinking through to understand it.
#
# __VARIABLE__ --> Refers to current number of tokens in each of the tgt sequences. 1 token gets added to
#                  each of the tgt sequences with 1 'advance' (method below) step.
#
# Assuming you already know about Beam Search, this is an implementation of the basic version of Beam Search for 
# Machine Translation. Lets go through the implementation logic at a high level before we dive into the code.
# 
# The input to beam search is a batch of source sequences -- [batch_size, source_sequence_length].
# Each 1D tensor in the batch represents a source sequence where each element is a token (from source vocabulary) 
# in the sequence.
#
# The output of beam search is a batch of target sequences (or translated sequence) -- [batch_size, target_sequence_length].
# Each 1D tensor in the batch represents a target sequence where each element is a token (from target vocabulary) 
# in the sequence.
#
# In Beam Search, at every position, instead of just predicting the token with the highest probability, we keep track
# of 'beam_width' number of tokens with the highest probability. So, We will have 'beam_width' number of potential 
# target sequences at each time step. The 'beam_width' is a hyperparameter that we need to set before running the
# beam search algorithm.
#
# Now going into the details, at each time step, each source sequence can have multiple potential target sequences 
# (or tokens) since we keep track 'beam_width' number of potential target sequences at each time step. So, 
# along with the target sequences, we need to keep track of the index of the source sequence this target sequence 
# belongs to in the batch. The state of each target sequence is stored in the 'SequenceState' (below in the code) 
# objects.
#
# During the beam search, some of the target sequences may reach the end of the sequence (<eos> token predicted 
# for that target sequence) before others. We need to keep track of the target sequences that have reached the end 
# of the sequence. We store the target sequences that have reached the end of the sequence in the 'complete_state' 
# dictionary (below in the code). The key of the dictionary is the index of the source sequence in the source batch 
# and the value is the 'SequenceState' object that has reached the end of the sequence (tgt) and has the highest 
# probability among all the sequences that reached the end. The list of target sequences that haven't reached the 
# end of the sequence are stored in the 'running_state' list.
#
# The beam search algorithm is iterated 'TGT_SEQ_LIMIT' number of times. At each iteration, we predict the next token
# in the target sequence for each target sequence in the 'running_state' list. The 'running_state' and 'complete_state'
# are updated based on the predicted tokens.
#
# The state update logic is as follows:
# 1) If the target sequence has reached the end of the sequence, then we store the target sequence in the 
#    'complete_state' if this target sequence has the highest probability among all the target sequences that have
#    reached the end of the sequence.
# 2) If the target sequence hasn't reached the end of the sequence, then this target sequence is a potential target
#    sequence to be considered for the next iteration. For each (source sequence, target sequence) pair, we retrieve
#    'beam_width' number of potential tokens with the highest probability from the model. We will now have a list of
#    target sequences for a specific source sequence. We will sort this list based on the probability of the target
#    sequences and keep only the 'beam_width' number of target sequences with the highest probability for the next
#    iteration.
# 3) If all the target sequences for a source sequence have reached the end of the sequence, then this source sequence
#    will not be considered for the next iteration.
# 
# Once we have iterated 'TGT_SEQ_LIMIT' number of times, we will have the 'complete_state' dictionary with the target
# sequences that have the highest probability for most (hopefully) of the source sequence in the batch. However, some 
# of the source sequences may not have any target sequences in the 'complete_state' dictionary. In this case, we will 
# consider the target sequence with the highest probability from the 'running_state' list that for these specific 
# source sequence and add them to the 'complete_state' dictionary.
#
# Finally, we will have the target sequences with the highest probability for each source sequence in the batch stored
# in the 'complete_state' dictionary. We will return these target sequences as the output of the beam search algorithm.
#
# Back tracking a bit, lets expand on the part where we have a 'running_state' and we need to update it based on the 
# predicted tokens. As noted above, the 'running_state' might contain multiple target sequences for a source sequence.
# We are passing these (source sequence, target sequence) pairs to the decoder part of the model to get the probability 
# distribution over the target vocabulary. So, we will have to construct the input to the decoder from the 
# 'running_state' list. The input to the decoder is a batch of source sequences and target sequences. So, we bascially 
# copy the source sequence as many times as it is used in the 'running_state' list and form the source sequence batch
# -- This might be a bit confusing, but you might understand (hopefully) it better when you see the code. src_mask is 
# also constructed in a similar way. The target sequence batch is constructed by taking the target sequences from the 
# 'running_state' list.
#
# Refer to 'step_19_beam_search.ipynb' (link to the notebook) to understand the code implementation with an example.

from dataclasses import dataclass

from model_implementation.data_processing.data_preparation.data_batching_and_masking import construct_look_ahead_mask
from model_implementation.model_building.machine_translation_model import MachineTranslationModel
from torch import Tensor
from typing import Dict, List, Tuple

import torch

# Contains the state of the sequence that is predicted by the beam search.
@dataclass
class SequenceState:
    # Index of the source sentence for which the current target sequence has been predicted.
    index: int
    # Sequence of tokens in the target prediction.
    tokens: Tensor
    # Log of the probability that this is the current translation for the source sequence.
    log_prob: float


# Holds the state of Beam search.
class BeamState:
    def __init__(self):
        # Holds one complete sequence (if found already) with the maximum probability for every source
        # sequence.
        self.complete_state: Dict[int, SequenceState] = {}
        # List of predicted tgt sequences for which the <eos> token has not been predicted yet. So,
        # these sequences are potentially incomplete and will continue through the prediction process.
        self.running_state: List[SequenceState] = []


class BeamSearch:
    def __init__(self, 
                 translation_model: MachineTranslationModel, 
                 tgt_seq_limit: int,
                 sos_token_id: int,
                 eos_token_id: int,
                 beam_width: int,
                 device: str):
        """Initialized the BeamSearch object.

        Args:
            translation_model (MachineTranslationModel): The translation model that is used for predictions.
            tgt_seq_limit (int): Maximum number of tokens that can be predicted for any target sequence. 
            sos_token_id (int): Token id for the start of the sentence token.
            eos_token_id (int): Token id for the end of the sentence token.
            beam_width (int): Number of potential target token predictions to be considered at each time step.
            device (str): Device to be used for storing the tensors. Can be either 'cpu' or 'cuda'.
        """
        self.translation_model = translation_model
        self.tgt_seq_limit = tgt_seq_limit
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.beam_width = beam_width
        self.device = device


    def decode(self, src_batch: Tensor, src_mask: Tensor) -> List[Tensor]:
        """Translates the source sentences to target sentences using beam search.

        Args:
            src_batch (Tensor): A batch of source sentences to be translated.
                                SHAPE: [batch_size, seq_len]
            src_mask (Tensor): Mask to prevent the padding tokens in the source sentences to attend 
                               to the actual tokens in the sources sentences.
                               SHAPE: [batch_size, 1, seq_len, seq_len]
        Returns:
            List[Tensor]: Translated target sentences (contains token ids in the target vocabulary).
                          Length of list: batch_size
                          Shape of the tensors within the list: 1D tensors with variable length. 
        """
        # Move the source batch and mask to the device. This is done in an effort move the tensors to GPU.
        # If the device is 'cpu', then the tensors will be stored on CPU.
        src_batch = src_batch.to(self.device)
        src_mask = src_mask.to(self.device)
        # Pass the source sentence through the encoder to find the encoded src sentence tokens.
        encoded_src = self.translation_model.encode(src=src_batch, src_mask=src_mask)
        # Initialize the running_state for beam search to start.
        self.__initialize_beams(src_batch.size(0))
        # We generate 'tgt_seq_limit' number of tokens (the maximum allowed) in the target sequences.
        for _ in range(self.tgt_seq_limit):
            # If there are no potential tgt sequences, we stop the beam search. This essentially means
            # we already found atleast 'beam_width' number of complete tgt sequences for each src
            # sequence in the batch.
            if len(self.beam_state.running_state) == 0:
                break
            # As we proceed forward in the algorithm, some of the src sequences might have already found have found <eos> 
            # token in all of its corresponding tgt sequences and so these should not be used anymore for further 
            # predictions. Also, there might multiple potential tgt sequences for a single src sequence and so we need
            # to copy the same src sequences multiple times. These are handled in the below function call.
            src_for_inference, src_mask_for_inference = self.get_src_for_running_state(encoded_src=encoded_src, src_mask=src_mask)
            # Gets the corresponding tgt sequences and theirs masks as a tensor to be used with the
            # translation model. 
            tgt_for_inference, tgt_mask_for_inference = self.get_tgt_for_running_state()
            # Predict 1 token for all the running tgt sequences and update the beam state.
            self.advance(encoded_src=src_for_inference, src_mask=src_mask_for_inference, tgt_batch=tgt_for_inference, tgt_mask=tgt_mask_for_inference)
        self.add_running_state_to_complete_state()
        return [self.beam_state.complete_state[src_seq_idx].tokens[1:-1] for src_seq_idx in sorted(self.beam_state.complete_state.keys())]


    def add_running_state_to_complete_state(self):
        """Adds the running sequences to complete sequences if there are no complete tgt sequences for 
        a source sequence."""
        dummy_tensor = torch.tensor(data=[self.sos_token_id], dtype=torch.int32, device=self.device)
        # Holds the sequence with the maximum probability for a given source sequence.
        max_prob_state = SequenceState(index=-1, tokens=dummy_tensor, log_prob=float('-inf'))
        for state in self.beam_state.running_state:
            # If the source sequence of the current running sequence is not already in the complete sequences, 
            # that means we haven't found a complete sequence for this (identified by the index) source sequence 
            # yet. So, we hold this sequence in the max_prob_state as a potential complete sequence.
            if state.index not in self.beam_state.complete_state:
                # If we haven't found any complete sequence yet, we just store this sequence as the potential
                # complete sequence in the max_prob_state.
                if max_prob_state.index == -1:
                    max_prob_state = state
                elif max_prob_state.index != state.index:
                    # If all the running sequences have been looked at for the previous src sequence (identified 
                    # by max_prob_state.index), then we just add the potential complete sequence to the list of
                    # complete sequences.
                    self.beam_state.complete_state[state.index] = max_prob_state
                    # We also store the new running sequence as a new potential running sequence for the new src
                    # sequence (identified by the index state.index).
                    max_prob_state = state
                else:
                    # If a new potential complete sequence is found, we pick the one with maximum probability and
                    # store it as the potential complete sequence.
                    if max_prob_state.log_prob < state.log_prob:
                        max_prob_state = state
        # Add the left out potential sequence to the list of complete sequences.
        if max_prob_state.index != -1:
            max_prob_state.tokens = torch.cat(tensors=[max_prob_state.tokens, torch.tensor(data=[self.eos_token_id], dtype=torch.int32, device=self.device)], dim=0)
            self.beam_state.complete_state[max_prob_state.index] = max_prob_state


    def get_src_for_running_state(self, encoded_src: Tensor, src_mask: Tensor) -> Tuple[Tensor, Tensor]:
        """Creates a new encoded src tensor in which the src sequences match the indices in the running 
        state maintained by beam search. Also, creates the src masks for the corresponding src sequences.

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
        src_indices = torch.tensor(data=[state.index for state in self.beam_state.running_state], dtype=torch.int32, device=self.device)
        # Select the tensors corresponding to the indices in 'src_indices' and create a new
        # tensor to be used as a src for inference.
        # Example src_for_inference: [0th tensor, 0th tensor, 3rd tensor, 3rd tensor, 3rd tensor, 7th tensor, 7th tensor, 7th tensor, 8th tensor] 
        src_for_inference = torch.index_select(input=encoded_src, dim=0, index=src_indices).to(self.device)
        # Create the src mask similarly.
        src_mask_for_inference = torch.index_select(input=src_mask, dim=0, index=src_indices).to(self.device)
        return src_for_inference, src_mask_for_inference


    def get_tgt_for_running_state(self) -> Tuple[Tensor, Tensor]:
        """Creates a new tgt tensor by stacking all the tgt sequences in the running state maintained 
        by beam search. Also, creates the tgt masks for the corresponding tgt sequences.

        Returns:
            Tuple[Tensor, Tensor]: tgt sequence batch tensor and the corresponding mask to be used with Decoder.
                                   SHAPE: ([len(running_state), __VARIABLE__, d_model], [len(running_state), 1, __VARIABLE__, __VARIABLE__])
        """
        tgt_for_inference = torch.stack(tensors=[state.tokens for state in self.beam_state.running_state], dim=0).to(self.device)
        # Creates look ahead mask and updates its shape to a 4D tensor as required by the Decoder.
        # Since there are no padding tokens in the tgt sequences, the same mask is going to be used by all
        # tgt sequences in the batch and so we just repeat the same mask len(running_state) number of times.
        tgt_mask_for_inference = construct_look_ahead_mask(size=tgt_for_inference.size(1)).unsqueeze(0).unsqueeze(0).repeat(len(self.beam_state.running_state), 1, 1, 1).to(self.device)
        return tgt_for_inference, tgt_mask_for_inference
    

    def __initialize_beams(self, batch_size: int):
        """Initializes the beam state and specifically the running state for each of src sequences.

        Args:
            batch_size (int): Number of src sequences to be translated to tgt language.
        """
        self.beam_state = BeamState()
        # The decoder needs the initial 'sos_token_id' token for the prediction process to start. So,
        # we create 1 tgt token sequence for the corresponding src sequence i.e., we create 1 
        # 'SequenceState' object for 1 tgt sequence in the running_state. Each 'SequenceState' object 
        # in the running state corresponds to one potential tgt sequence the beam search algorithm is 
        # looking to expand by predicting next tokens.  
        self.beam_state.running_state = [SequenceState(index=idx, tokens=torch.tensor(data=[self.sos_token_id], dtype=torch.int32, device=self.device), log_prob=0.0) for idx in range(batch_size)]


    def __update_beam_state(self, predicted_log_probabilities: Tensor):
        """Updates the state of the beam search by appending the newly predicted tokens to the running tgt 
        sequences. If a tgt sequence is complete (<eos> token is predicted), it is added (depends on log_prob) 
        to the complete sequences. If the tgt sequence is not complete, it is added back to the running state 
        for the next iteration (to predict one more token).

        Args:
            predicted_log_probabilities (Tensor): The probability distributions over the tgt vocabulary for 
                                                  each of the tokens in each of the current tgt sequences.
                                                  SHAPE: [len(running_state), __VARIABLE__, TGT_VOCAB_SIZE]
        """
        # Extracts the probabilities only for the last predicted token. We don't care about the tokens
        # that are in positions where the model already predicted the output tokens. We only care about
        # the last token that needs to be appended to the existing sequence to extend it.
        predicted_log_probabilities = predicted_log_probabilities[:, -1, :]
        # A given source sentence can have multiple potential target sequences if beam_width > 1. Here, 
        # we find the number of target sequences currently being used (in running_state) to predict the 
        # next token for each of the source sentences.
        _, tgt_group_counts = torch.unique(input=torch.tensor(data=[seq_state.index for seq_state in self.beam_state.running_state], dtype=torch.int16), return_counts=True)
        # Holds the running_state for the next iteration of beam search.
        new_running_state: List[SequenceState] = []
        # Index to keep track of the start of the group of tgt sequences for a single source sequence.
        start_index = 0
        for tgt_group_size in tgt_group_counts:
            # Extract the group of SequenceState objects for a single source sentence.
            old_tgt_state_group = self.beam_state.running_state[start_index: start_index + tgt_group_size.item()]
            # Extract the group of probabilities for the corresponding tgt sequence token predictions.
            new_tgt_prob_group = torch.unbind(predicted_log_probabilities[start_index: start_index + tgt_group_size.item()], dim=0)
            # Holds all the new sequences formed after appending the token for the previous group of tgt sequences.
            running_beams: List[SequenceState] = []
            # Holds all the sequences for which the <eos> token has been predicted.
            complete_beams: List[SequenceState] = []
            # Iterate on the old SequenceState object and the corresponding next prediction probabilities for this 
            # tgt sequence.
            for old_seq_state, new_tgt_pred_probs in (zip(old_tgt_state_group, new_tgt_prob_group)):
                # Index of the source sentence for which the translations are being calculated.
                src_seq_idx = old_seq_state.index
                # Extract the top few tokens to be considered as the next token via beam search.
                top_probs, top_tokens = new_tgt_pred_probs.topk(k=self.beam_width, dim=-1)
                # Iterate on each predicted token, create the sequence with this token appended and calculate
                # the probability of the new sequence (with token appended).
                for pred_prob, pred_token in zip(top_probs, top_tokens):
                    # Append the newly predicted token to the existing sequence of tokens.
                    updated_token_seq = torch.cat(tensors=[old_seq_state.tokens, pred_token.unsqueeze(0).to(torch.int32)])
                    # The log probability of the extended sequence is the probability of the old sequence added
                    # to the probability associated with the newly predicted token.
                    updated_seq_prob = old_seq_state.log_prob + pred_prob.item()
                    # Creates a new SequenceState object associated with the extended tgt sequence.
                    new_state = SequenceState(index=src_seq_idx, tokens=updated_token_seq, log_prob=updated_seq_prob)
                    if pred_token.item() == self.eos_token_id:
                        # If the newly predicted token is <eos>, then this tgt sequence is complete and we add it
                        # to the list of complete sequences for this specific src sequence.
                        complete_beams.append(new_state)
                    else:
                        # If the newly predicted token is not <eos>, then this tgt sequence is not complete and
                        # can be extended by predicting further tokens.
                        running_beams.append(new_state)
            # If the newly predicted token is an <eos> token, then we remove these tgt sequences from the 
            # beam search and update the complete state for the corresponding src sequence accordingly.
            if len(complete_beams) > 0:
                # Index of the source sentence for which the translations are being calculated.
                src_seq_idx = complete_beams[0].index
                # sort the completed sequences according to their probabilities in descending order. 
                complete_beams.sort(key=lambda seq_state: seq_state.log_prob, reverse=True)
                if src_seq_idx in self.beam_state.complete_state:
                    # If we found complete sequences before for this specific source sequence, we only store
                    # the complete sequence for which the probability of occurence is the highest.
                    if self.beam_state.complete_state[src_seq_idx].log_prob < complete_beams[0].log_prob:
                        self.beam_state.complete_state[src_seq_idx] = complete_beams[0]
                else:
                    # If this is the first complete sequence we found, we just store this specific sequence.
                    self.beam_state.complete_state[src_seq_idx] = complete_beams[0]           
            if len(running_beams) > 0:              
                # sort the running sequences according to their probabilities in descending order.
                running_beams.sort(key=lambda seq_state: seq_state.log_prob, reverse=True)
                # Add the running sequences for further predictions. Only add the first 'beam_width' number
                # of sequences.
                new_running_state.extend(running_beams[:min(self.beam_width, len(running_beams))])
                # Update the start_index to the start of the next group of tgt sequences.
            start_index += tgt_group_size.item()
        self.beam_state.running_state = new_running_state


    def advance(self, encoded_src: Tensor, src_mask: Tensor, tgt_batch: Tensor, tgt_mask: Tensor):
        """Predicts 1 token for all the (src, tgt) pairs and appends it at the back of each
        tgt sequence. Also, updates the beam state according the newly predicted tokens and their
        associated probabilities.

        Args:
            encoded_src (Tensor): Encoded source sequences i.e., the output of the Encoder after passing the src_batch.
                                  SHAPE: [len(running_state), seq_len, d_model]
            src_mask (Tensor): Mask for the corresponding encoded_src.
                               SHAPE: [len(running_state), 1, seq_len, seq_len]
            tgt_batch (Tensor): The input tgt sequences for which a new token needs to be predicted.
                                SHAPE: [len(running_state), __VARIABLE__, d_model]
            tgt_mask (Tensor): Mask for the corresponding tgt_batch.
                               SHAPE: [len(running_state), 1, __VARIABLE__, __VARIABLE__]
        """
        # Run the decoder on the src and the corresponding running tgt sequences.
        decoded_tgt = self.translation_model.decode(tgt=tgt_batch, tgt_mask=tgt_mask, encoded_src=encoded_src, src_mask=src_mask)
        # Convert the decoder output into token probabilties.
        predicted_log_probabilities = self.translation_model.token_predictor(decoded_tgt)
        # Update the beam state according the predicted tokens for each of the tgt sequences.
        self.__update_beam_state(predicted_log_probabilities=predicted_log_probabilities)