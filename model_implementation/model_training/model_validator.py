# This file implements the validation of the model during training. The validation is done after every epoch
# and the loss per token in the validation datasets is logged. Ideally, this should have been implemented as 
# part of the model training loop but we have separated it out for better readability and lack of good 
# planning at the beginning of the project.

from model_implementation.data_processing.data_preparation.dataset_wrapper import DatasetWrapper
from model_implementation.data_processing.data_preparation.data_batching_and_masking import Batch
from model_implementation.data_processing.data_preparation.data_loader import create_data_loader
from model_implementation.data_processing.tokenization.base_tokenizer import BaseTokenizer
from model_implementation.model_building.machine_translation_model import MachineTranslationModel
from model_implementation.model_training.label_smoothing import LabelSmoothing
from model_implementation.model_training.loss_computation import LossCompute
from model_implementation.utils.constants import BATCH_SIZE, NUM_WORKERS, SMOOTHING_PROB, MAX_INPUT_SEQUENCE_LENGTH
from model_implementation.utils.logger import get_logger
from torch.utils.data import DataLoader


logger = get_logger(__name__)


def compute_loss_per_token(machine_translation_model: MachineTranslationModel, 
                           validation_dataLoader: DataLoader, 
                           loss_compute: LossCompute,
                           label_smoothing: LabelSmoothing,
                           pad_token_id: int,
                           device: str) -> float:
    """Runs the training loop on the entire dataset and saves the trained model to disk.

    Args:
        machine_translation_model (MachineTranslationModel): The translation model being trained.
        validation_dataLoader (DataLoader): DataLoader containing the batched English - Telugu translation data for validation.
        loss_compute (LossCompute): Computes the KL Divergence loss between the prediction and target probability 
                                    distributions.
        label_smoothing (LabelSmoothing): Applies label smoothing to the target token probability distributions. 
                                          Prevents the model from becoming over confident or over-fitting on data.
        pad_token_id (int): Id of the padding token. Usually set to 2.
        device (str): Device to be used for training the model. Can be either 'cpu' or 'cuda'.

    Returns:
        float: Returns the loss per token in the validation dataset.
    """
    total_loss = 0
    num_tokens_processed = 0
    num_batches_processed = 0
    for src_batch, tgt_batch in validation_dataLoader:
        if src_batch.shape[1] > MAX_INPUT_SEQUENCE_LENGTH or tgt_batch.shape[1] > MAX_INPUT_SEQUENCE_LENGTH:
            logger.debug(f"Sequence length of src_batch: {src_batch.shape[1]} and Sequence length of tgt_batch: {tgt_batch.shape[1]}")
            logger.debug(f"Skipping batch {num_batches_processed} as the sequence length is greater than the maximum sequence length.")
            continue
        # Batch size is a relatively small number (32). So, to avoid spamming, we print the updates once
        # for every 100 batches.
        if num_batches_processed % 100 == 0:
            logger.info(f"Processing validation batch number: {num_batches_processed}")
        # Create the Batch from the input.
        batch = Batch(src_batch=src_batch, tgt_batch=tgt_batch, pad_token_id=pad_token_id, device=device)
        num_tokens_processed += int(batch.non_pad_tokens.item())
        # Forward pass of the machine translation model. Returns the predicted probability distributions for each token
        # in the target sentences in the tgt_batch.
        predicted_tgt_log_probability_distributions = machine_translation_model(src=batch.src, 
                                                                                tgt=batch.tgt_decoder_input, 
                                                                                src_mask=batch.src_mask, 
                                                                                tgt_mask=batch.tgt_mask)
        # Applies label smoothing to the expected token ids.
        expected_tgt_probability_distributions = label_smoothing(targets=batch.tgt_expected_decoder_output)
        # Computes the KLDivergence loss between predicted and expected outputs.
        loss_per_token_in_batch = loss_compute(log_predictions=predicted_tgt_log_probability_distributions, 
                                               targets=expected_tgt_probability_distributions, 
                                               num_non_pad_tokens=int(batch.non_pad_tokens.item()))
        total_loss += (loss_per_token_in_batch.item() * int(batch.non_pad_tokens.item()))
        # Update the number of batches processed.
        num_batches_processed += 1
    return (total_loss / num_tokens_processed)


def validate_model(translation_model: MachineTranslationModel,
                   validation_dataset: DatasetWrapper,
                   english_tokenizer: BaseTokenizer, 
                   telugu_tokenizer: BaseTokenizer, 
                   pad_token_id: int,
                   device: str) -> float:
    """Runs the Machine translation model on the given validation dataset and returns the validation loss.

    Args:
        english_tokenizer (BaseTokenizer): English tokenizer to tokenize the English sentences.
        telugu_tokenizer (BaseTokenizer): Telugu tokenizer to tokenize the Telugu sentences.
        pad_token_id (int): Id of the padding token. Usually set to 2.
        device (str): Device to be used for training the model. Can be either 'cpu' or 'cuda'.

    Returns:
        float: Returns the loss per token in the validation dataset.
    """
    logger.info("Running the model on the validation dataset.")
    # Set the model to evaluation mode so that the model does not update the weights during validation.
    translation_model.eval()
    # Get the vocabulary sizes for the target language.
    tgt_vocab_size = telugu_tokenizer.get_vocab_size()
    # Label smoothing to prevent the model from becoming over confident or over-fitting on data.
    label_smoothing = LabelSmoothing(tgt_vocab_size=tgt_vocab_size, padding_idx=pad_token_id, smoothing=SMOOTHING_PROB, device=device)
    # Move the label smoothing to the device (gpu in our case) to train the model on gpu. If gpu is not available, the model
    # will be trained on the cpu.
    label_smoothing.to(device)
    # Loss computation to calculate the KL Divergence loss between the predicted and target probability distributions.
    loss_compute = LossCompute()
    # Create the DataLoader to load the training data.
    validation_dataloader = create_data_loader(dataset=validation_dataset, 
                                               english_tokenizer=english_tokenizer, 
                                               telugu_tokenizer=telugu_tokenizer, 
                                               num_workers=NUM_WORKERS, 
                                               batch_size=BATCH_SIZE)
    loss_per_token = compute_loss_per_token(machine_translation_model=translation_model, 
                                            validation_dataLoader=validation_dataloader, 
                                            loss_compute=loss_compute, 
                                            label_smoothing=label_smoothing,  
                                            pad_token_id=pad_token_id,
                                            device=device)
    logger.info(f"Loss per token in the validation dataset: {loss_per_token}")
    # Return the loss per token in the validation dataset.
    return loss_per_token