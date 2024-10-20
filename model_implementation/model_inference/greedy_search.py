
from model_implementation.data_processing.data_preparation.data_batching_and_masking import construct_look_ahead_mask
from model_implementation.model_building.machine_translation_model import MachineTranslationModel
from model_implementation.utils.constants import MAX_INFERENCE_SEQ_LEN
from model_implementation.utils.logger import get_logger
from torch import Tensor

import torch


logger = get_logger(name=__name__)


def update_tgt_batch(tgt_batch: Tensor, predicted_log_probabilities: Tensor) -> Tensor:
    predicted_log_probabilities = predicted_log_probabilities[:, -1, :]
    _, predicted_tokens = predicted_log_probabilities.max(dim=1, keepdim=True)
    return torch.cat((tgt_batch, predicted_tokens), dim=1)


def greedy_search(translation_model: MachineTranslationModel, 
                  src_batch: Tensor, 
                  src_mask: Tensor, 
                  sos_token_id: int, 
                  eos_token_id: int, 
                  device: str) -> Tensor:
    batch_size = src_batch.size(0)
    # Pass the source sentence through the encoder to find the encoded src sentence tokens.
    encoded_src = translation_model.encode(src=src_batch, src_mask=src_mask)
    tgt_batch = torch.tensor(data=[[sos_token_id for _ in range(batch_size)]], dtype=torch.int32, device=device)
    tgt_mask = construct_look_ahead_mask(size=tgt_batch.size(1)).unsqueeze(0).unsqueeze(0).to(device)
    for _ in range(MAX_INFERENCE_SEQ_LEN):
        # Run the decoder on the src and the corresponding running tgt sequences.
        decoded_tgt = translation_model.decode(tgt=tgt_batch, tgt_mask=tgt_mask, encoded_src=encoded_src, src_mask=src_mask)
        # Convert the decoder output into token probabilties.
        predicted_log_probabilities = translation_model.token_predictor(decoded_tgt)
        tgt_batch = update_tgt_batch(tgt_batch=tgt_batch, predicted_log_probabilities=predicted_log_probabilities)
        tgt_mask = construct_look_ahead_mask(size=tgt_batch.size(1)).unsqueeze(0).unsqueeze(0).to(device)
        if tgt_batch[0][-1] == eos_token_id:
            break
    logger.info(f"predictions: ", tgt_batch)
    return tgt_batch[:, 1:-1]