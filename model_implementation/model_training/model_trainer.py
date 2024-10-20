# This file implements the training loop for the model on the provided dataset. The training loop is run for the
# specified number of epochs and the model is saved to disk after every epoch.

from platform import machine
from model_implementation.data_processing.data_preparation.dataset_wrapper import DatasetWrapper
from model_implementation.data_processing.data_preparation.data_batching_and_masking import Batch
from model_implementation.data_processing.data_preparation.data_loader import create_data_loader
from model_implementation.data_processing.tokenization.base_tokenizer import BaseTokenizer
from model_implementation.model_building.machine_translation_model import MachineTranslationModel
from model_implementation.model_training.label_smoothing import LabelSmoothing
from model_implementation.model_training.learning_rate_schedule import rate
from model_implementation.model_training.loss_computation import LossCompute
from model_implementation.model_training.model_validator import validate_model
from model_implementation.utils.constants import (
    BATCH_SIZE, BETA_1, BETA_2, DROPOUT_PROB, D_FEED_FORWARD, D_MODEL, EPSILON, INITIAL_LEARNING_RATE, 
    MAX_INPUT_SEQUENCE_LENGTH, NUM_LAYERS, NUM_HEADS, NUM_WARMUP_STEPS, NUM_WORKERS, SMOOTHING_PROB
)
from model_implementation.utils.helpers import get_full_model_path_from_name, load_model_from_disk, save_model_to_disk
from model_implementation.utils.logger import get_logger
from model_implementation.utils.state_holders import ModelState, TrainState
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Tuple

import time
import torch

logger = get_logger(__name__)


def train_model_on_epoch(machine_translation_model: MachineTranslationModel, 
                         train_dataLoader: DataLoader, 
                         optimizer: torch.optim.Optimizer, 
                         lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
                         loss_compute: LossCompute,
                         label_smoothing: LabelSmoothing,
                         epoch_num: int,
                         pad_token_id: int,
                         device: str) -> TrainState:
    """Runs the training loop on the entire dataset and saves the trained model to disk.

    Args:
        machine_translation_model (MachineTranslationModel): The translation model being trained.
        train_dataLoader (DataLoader): DataLoader containing the batched English - Telugu translation data.
        optimizer (torch.optim.Optimizer): Optimizer used to update the weights of the model.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler that adjusts the learning rate as
                                                              the training progresses.
        loss_compute (LossCompute): Computes the KL Divergence loss between the prediction and target probability 
                                    distributions.
        label_smoothing (LabelSmoothing): Applies label smoothing to the target token probability distributions. 
                                          Prevents the model from becoming over confident or over-fitting on data.
        epoch_num (int): The current epoch i.e., the current iteration on the entire dataset.
        pad_token_id (int): Id of the padding token. Usually set to 2.
        device (str): Device to be used for training the model. Can be either 'cpu' or 'cuda'.

    Returns:
        TrainState: Returns the train state of the model after the current epoch.
    """
    # Time in seconds elapsed since the beginning of the epoch (Not training epoch but the standard point from which time is calculated).
    # This is the time at which the training of the model on the current epoch started.
    start_time = time.time()
    # Set the model to training mode so that Dropout and Batch Normalization are resumed.
    machine_translation_model.train()
    # Holds the learning rate at the start of the epoch.
    epoch_start_learning_rate = lr_scheduler.get_last_lr()[0]
    num_tokens_processed = 0
    num_batches_processed = 0
    num_batches_skipped = 0
    total_loss = 0.0
    for src_batch, tgt_batch in tqdm(train_dataLoader):
        if src_batch.shape[1] > MAX_INPUT_SEQUENCE_LENGTH or tgt_batch.shape[1] > MAX_INPUT_SEQUENCE_LENGTH:
            logger.debug(f"Sequence length of src_batch: {src_batch.shape[1]} and Sequence length of tgt_batch: {tgt_batch.shape[1]}")
            logger.debug(f"Skipping batch {num_batches_processed} as the sequence length is greater than the maximum sequence length.")
            num_batches_skipped += 1
            continue
        # Batch size is a relatively small number (32). So, to avoid spamming, we print the updates once
        # for every 100 batches.
        if num_batches_processed % 100 == 0:
            logger.info(f"Processing batch number: {num_batches_processed}")
        # Create the Batch from the input.
        batch = Batch(src_batch=src_batch, tgt_batch=tgt_batch, pad_token_id=pad_token_id, device=device)
        num_tokens_processed += int(batch.non_pad_tokens.item())
        logger.debug(f"POINT 0 -- shape of src: {batch.src.shape}, shape of src_mask: {batch.src_mask.shape}")
        logger.debug(f"POINT 1 -- shape of tgt_decoder_input: {batch.tgt_decoder_input.shape}, shape of tgt_mask: {batch.tgt_mask.shape},  shape of tgt_expected_decoder_output: {batch.tgt_expected_decoder_output.shape}")
        # Forward pass of the machine translation model. Returns the predicted probability distributions for each token
        # in the target sentences in the tgt_batch.
        predicted_tgt_log_probability_distributions = machine_translation_model(src=batch.src, 
                                                                                tgt=batch.tgt_decoder_input, 
                                                                                src_mask=batch.src_mask, 
                                                                                tgt_mask=batch.tgt_mask)
        # Applies label smoothing to the expected token ids.
        expected_tgt_probability_distributions = label_smoothing(targets=batch.tgt_expected_decoder_output)
        # Computes the KLDivergence loss between predicted and expected outputs.
        loss = loss_compute(log_predictions=predicted_tgt_log_probability_distributions, 
                            targets=expected_tgt_probability_distributions, 
                            num_non_pad_tokens=int(batch.non_pad_tokens.item()))
        # Computes the gradients wrt to the loss.
        loss.backward()
        # Updates the weights with the calculated gradients.
        optimizer.step()
        # Updates the learning rate. Notice that we are updating the learning rate after every batch of training
        # and not after every epoch. So, the 'epoch' value in lr_scheduler is not the same as the epoch in general
        # which is the number of steps the entrire training set is trained on.
        lr_scheduler.step()
        # zero out the gradients after the update so that the memory is wiped clean at the end of the epoch.
        optimizer.zero_grad(set_to_none=True)
        # Update the number of batches processed.
        num_batches_processed += 1
        total_loss += (loss.item() * int(batch.non_pad_tokens.item())) 
    # Time at which the training of the model on the current epoch ended.
    end_time = time.time()
    epoch_training_time = (end_time - start_time) / 60
    train_state = TrainState(epoch_num=epoch_num, 
                             training_time_in_minutes=epoch_training_time,
                             epoch_start_learning_rate=epoch_start_learning_rate,
                             epoch_end_learning_rate=lr_scheduler.get_last_lr()[0], 
                             training_loss=(total_loss / num_tokens_processed), 
                             num_tokens_processed=num_tokens_processed, 
                             model_checkpoint_path="",
                             num_batches_skipped=num_batches_skipped)
    return train_state


def train_model(num_epochs: int, 
                translation_dataset: DatasetWrapper,
                validation_dataset: DatasetWrapper,
                english_tokenizer: BaseTokenizer, 
                telugu_tokenizer: BaseTokenizer, 
                pad_token_id: int,
                model_name: str,
                model_checkpoint_prefix: str,
                device: str,
                resume_training: Optional[bool]=False) -> Tuple[MachineTranslationModel, ModelState]:
    """Trains the Machine translation model on the given dataset for the specified number of epochs.

    Args:
        num_epochs (int): Number of epochs i.e., the number of times the model is trained on the entire dataset.
        translation_dataset (DatasetWrapper): Dataset containing the English - Telugu translation data.
        validation_dataset (DatasetWrapper): Dataset containing the validation data to validate the model after each epoch.
        english_tokenizer (BaseTokenizer): English tokenizer to tokenize the English sentences.
        telugu_tokenizer (BaseTokenizer): Telugu tokenizer to tokenize the Telugu sentences.
        pad_token_id (int): Id of the padding token. Usually set to 2.
        model_name (str): Name to use to save the final model on disk.
        model_checkpoint_prefix (str): Prefix to be appended to the model_name while saving to disk.
        device (str): Device to be used for training the model. Can be either 'cpu' or 'cuda'.
        resume_training (bool): Flag to indicate if the training should be resumed for an existing model.

    Returns:
        Tuple[MachineTranslationModel, ModelState]: Returns the trained machine translation model and 
                                                    the model state.
    """
    # Get the vocabulary sizes for the source and target languages.
    src_vocab_size = english_tokenizer.get_vocab_size()
    tgt_vocab_size = telugu_tokenizer.get_vocab_size()
    # Initialize the Machine Translation model.
    translation_model = MachineTranslationModel(d_model=D_MODEL, 
                                                d_feed_forward=D_FEED_FORWARD,
                                                dropout_prob=DROPOUT_PROB, 
                                                num_heads=NUM_HEADS, 
                                                src_vocab_size=src_vocab_size, 
                                                tgt_vocab_size=tgt_vocab_size, 
                                                num_layers=NUM_LAYERS, 
                                                max_seq_len=MAX_INPUT_SEQUENCE_LENGTH)
    if resume_training:
        # Load an existing model state from disk to retrain if 'resume_training' is set to True.
        load_model_from_disk(model=translation_model, model_name=model_name, checkpoint_prefix=model_checkpoint_prefix)
    # Move the model to the device (gpu in our case) to train the model on gpu. If gpu is not available, the model
    # will be trained on the cpu.
    translation_model = translation_model.to(device)
    # Initialize the optimizer and the learning rate scheduler.
    adam_optimizer = torch.optim.Adam(params=translation_model.parameters(), lr=INITIAL_LEARNING_RATE, betas=(BETA_1, BETA_2), eps=EPSILON)
    # Lambda function to calculate the learning rate at each step. The learning rate is adjusted based on the number of
    # warmup steps and the current step number.
    rate_lambda = lambda step: rate(step, d_model=D_MODEL, warmup_steps=NUM_WARMUP_STEPS, factor=1.0)
    # Learning rate scheduler that adjusts the learning rate as the training progresses.
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=adam_optimizer, lr_lambda=rate_lambda)
    logger.debug(f"POINT 0 -- device being used to train the model: {device}")
    # Label smoothing to prevent the model from becoming over confident or over-fitting on data.
    label_smoothing = LabelSmoothing(tgt_vocab_size=tgt_vocab_size, padding_idx=pad_token_id, smoothing=SMOOTHING_PROB, device=device)
    # Move the label smoothing to the device (gpu in our case) to train the model on gpu. If gpu is not available, the model
    # will be trained on the cpu.
    label_smoothing.to(device)
    # Loss computation to calculate the KL Divergence loss between the predicted and target probability distributions.
    loss_compute = LossCompute()
    # Holds the state of the model after each epoch.
    model_state = ModelState(device=device)
    # Create the DataLoader to load the training data.
    train_dataloader = create_data_loader(dataset=translation_dataset, 
                                          english_tokenizer=english_tokenizer, 
                                          telugu_tokenizer=telugu_tokenizer, 
                                          num_workers=NUM_WORKERS, 
                                          batch_size=BATCH_SIZE)
    # Train the model for the specified number of epochs.
    for epoch in tqdm(range(num_epochs), desc="Training the model on epochs"):
        train_state = train_model_on_epoch(machine_translation_model=translation_model, 
                                           train_dataLoader=train_dataloader, 
                                           optimizer=adam_optimizer, 
                                           lr_scheduler=lr_scheduler, 
                                           loss_compute=loss_compute, 
                                           label_smoothing=label_smoothing, 
                                           epoch_num=epoch, 
                                           pad_token_id=pad_token_id,
                                           device=device)
        # Calculate the loss per token in the validation dataset.
        validation_loss = validate_model(translation_model=translation_model, 
                                         validation_dataset=validation_dataset, 
                                         english_tokenizer=english_tokenizer, 
                                         telugu_tokenizer=telugu_tokenizer, 
                                         pad_token_id=pad_token_id, 
                                         device=device)
        # Update the validation loss in the train_state.
        train_state.validation_loss = validation_loss
        # Name of the model checkpoint to be saved to disk.
        model_checkpoint_name = f"epoch_{epoch}_{model_name}"
        # Update the model_checkpoint_path to the correct path in the TrainState.
        train_state.model_checkpoint_path = get_full_model_path_from_name(model_name=model_checkpoint_name, checkpoint_prefix=model_checkpoint_prefix)
        # Log the state of the model training to the console for debugging purposes.
        train_state.log_state()
        # Add the train_state from the current epoch to the overall model state.
        model_state.append_state(train_state)
        # Save the model to disk in between epochs in case the training is interrupted and needs to be resumed.    
        save_model_to_disk(model=translation_model, 
                           model_name=model_checkpoint_name, 
                           checkpoint_prefix=model_checkpoint_prefix)
        remaining_training_time_in_minutes = (num_epochs - (epoch + 1)) * train_state.training_time_in_minutes
        logger.info(f"Approximate remaining training time in minutes: {remaining_training_time_in_minutes}")
    # Return the trained model and the model state after training.
    return translation_model, model_state