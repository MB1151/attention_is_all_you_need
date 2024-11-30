# This file implements the logic to translate the source sentences to the target language using the trained
# machine translation model. It uses the Beam Search algorithm to generate the target sequences.

from model_implementation.data_processing.data_preparation.data_batching_and_masking import construct_padding_mask
from model_implementation.data_processing.tokenization.base_tokenizer import BaseTokenizer
from model_implementation.model_building.machine_translation_model import MachineTranslationModel
from model_implementation.model_inference.beam_search import BeamSearch
from model_implementation.model_inference.greedy_search import GreedySearch
from model_implementation.utils.constants import START_TOKEN, END_TOKEN, PAD_TOKEN, MAX_INFERENCE_SEQ_LEN
from model_implementation.utils.logger import get_logger
from torch import Tensor
from typing import List, Tuple

import torch


logger = get_logger(__name__)


def equalize_src_seq_lengths(tokenized_src_sequences: List[List[int]], src_tokenizer: BaseTokenizer):
    """Equalizes the lengths of the source sequences by padding the shorter sequences with the pad token id.
    Does the padding in-place.
    
    Args:
        tokenized_src_sequences (List[List[int]]): A list of tokenized source sequences.
    """
    pad_token_id = src_tokenizer.get_token_id(PAD_TOKEN)
    max_src_seq_len = max([len(seq) for seq in tokenized_src_sequences])
    for src_seq in tokenized_src_sequences:
        src_seq.extend([pad_token_id] * (max_src_seq_len - len(src_seq)))


def create_src_input(src_sentences: List[str], src_tokenizer: BaseTokenizer, device: str) -> Tuple[Tensor, Tensor]:
    """Tokenizes the source sentences and creates the source batch and mask to be used as input to the 
    Translation model.

    Args:
        src_sentences (List[str]): List of source sentences (english language).
        src_tokenizer (BaseTokenizer): Tokenizer for the source language (english tokenizer).
        device (str): Device to be used for storing the tensors. Can be either 'cpu' or 'cuda'.
        
    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the source batch and source mask in that order.
                               SHAPE OF src_batch: [batch_size, seq_len]
                               SHAPE OF src_mask: [batch_size, 1, 1, seq_len]
    """
    # Holds the tokenized src sequences. Example: [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
    tokenized_src_sequences: List[List[int]] = []
    # Converts each source sentence into a list of token ids.
    for sentence in src_sentences:
        tokenized_src_sequences.append(src_tokenizer.encode(sentence))
    print(f"tokenized_src_sequences: {tokenized_src_sequences}")
    # The source sequences after tokenization could be of different lengths. We need to equalize the lengths
    # of the source sequences by padding the shorter sequences with the pad token id.
    equalize_src_seq_lengths(tokenized_src_sequences=tokenized_src_sequences, src_tokenizer=src_tokenizer)
    # Convert the tokenized source sequences to a tensor.
    # Shape of src_batch: [batch_size, seq_len]
    src_batch = torch.tensor(data=tokenized_src_sequences, dtype=torch.int32, device=device)
    # Construct the padding mask for the source sequences.
    # Shape of src_mask: [batch_size, 1, 1, seq_len]
    src_mask = construct_padding_mask(input=src_batch, pad_token_id=src_tokenizer.get_token_id(PAD_TOKEN)).unsqueeze(1).to(device)
    return src_batch, src_mask


def translate(translation_model: MachineTranslationModel, 
              src_tokenizer: BaseTokenizer, 
              tgt_tokenizer: BaseTokenizer, 
              src_sentences: List[str],
              beam_size: int,
              search_type: str,
              device: str) -> List[str]:
    """Translates the source sentences to the target language using the given translation model.
    Uses the Beam Search algorithm to generate the target sequences.

    Args:
        translation_model (MachineTranslationModel): Trained machine translation model (english to Telugu).
        src_tokenizer (BaseTokenizer): Tokenizer for the source language (english tokenizer).
        tgt_tokenizer (BaseTokenizer): Tokenizer for the target language (telugu tokenizer).
        src_sentences (List[str]): List of source sentences (english language).
        beam_size (int): Width of the beam to be used in the Beam Search algorithm.
        search_type (str): Type of search to be used to find the predictions. Can be either 'greedy' or 'beam'.
        device (str): Device to be used for storing the tensors. Can be either 'cpu' or 'cuda'.

    Returns:
        List[str]: List of translated sentences (telugu language).
    """
    # Create the source batch and mask which will be used as input to the model.
    src_batch, src_mask = create_src_input(src_sentences=src_sentences, src_tokenizer=src_tokenizer, device=device)
    print(f"src_batch: {src_batch}")
    print("-" * 150)
    print(f"src_mask: {src_mask}")
    sos_token_id = src_tokenizer.get_token_id(START_TOKEN)
    eos_token_id = src_tokenizer.get_token_id(END_TOKEN)
    if search_type == "beam":
        # Create an instance of the BeamSearch class.
        beam_search = BeamSearch(translation_model=translation_model, 
                                 tgt_seq_limit=MAX_INFERENCE_SEQ_LEN, 
                                 sos_token_id=sos_token_id, 
                                 eos_token_id=eos_token_id, 
                                 beam_width=beam_size,
                                 device=device)
        tokenized_tgt_sequences = beam_search.decode(src_batch=src_batch, src_mask=src_mask)
    elif search_type == "greedy":
        greedy_search = GreedySearch(translation_model=translation_model, 
                                     tgt_seq_limit=MAX_INFERENCE_SEQ_LEN, 
                                     sos_token_id=sos_token_id, 
                                     eos_token_id=eos_token_id, 
                                     device=device)
        tokenized_tgt_sequences = greedy_search.decode(src_batch=src_batch, src_mask=src_mask)
    else:
        raise ValueError(f"Invalid search type '{search_type}' for inference.")
    # Convert the target token ids to the corresponding tokens.
    translated_sentences = []
    # Converting each of the tokenized target sequences to the corresponding sentences (Telugu sentences).
    for tgt_seq in tokenized_tgt_sequences:
        translated_sentences.append(tgt_tokenizer.decode(tgt_seq.tolist()))
    return translated_sentences