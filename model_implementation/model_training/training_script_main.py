# This file implements the training script for the machine translation model. This script accepts one
# command line argument which is the type of tokenizer to be used. The script then loads the training
# dataset from disk, wraps the dataset in a pytorch Dataset, gets the tokenizers for the English and
# Telugu languages, and trains the translation model.

from model_implementation.data_processing.data_preparation.dataset_wrapper import DatasetWrapper
from model_implementation.data_processing.data_preparation.data_helpers import get_tokenizers, load_data_from_disk
from model_implementation.data_processing.tokenization.base_tokenizer import BaseTokenizer
from model_implementation.model_training.model_trainer import train_model
from model_implementation.utils.config import LOG_LEVEL
from model_implementation.utils.constants import (
    DEBUG_DATASET_PATH, ENGLISH_VOCAB_SIZE, FULL_EN_TE_DATASET_PATH, MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT, 
    MEMORY_SNAPSHOT_PATH, NUM_EPOCHS, PAD_TOKEN, TELUGU_VOCAB_SIZE, TRAIN_DATASET_PATH, 
    VALIDATION_DATASET_PATH, DEVICE_GPU
)
from model_implementation.utils.logger import get_logger
from model_implementation.utils.helpers import get_absolute_path, get_device, save_model_to_disk

import argparse
import datasets
import logging
import time
import torch

logging.basicConfig(level=LOG_LEVEL)
logger = get_logger(__name__)


def train_translation_model(device: str, 
                            model_name: str, 
                            model_checkpoint_prefix: str="",
                            tokenizer_type: str="bpe",
                            max_en_vocab_size: int=ENGLISH_VOCAB_SIZE,
                            max_te_vocab_size: int=TELUGU_VOCAB_SIZE,
                            retrain_tokenizers: bool=False,
                            resume_training: bool=False):
    """Trains the translation model and saves the trained model to disk with the given model name 
    and checkpoint prefix.

    Args:
        device (str): Device to be used for training the model. It can be either 'cpu' or 'cuda'.
        model_name (str): Name to use to save the final model on disk.
        model_checkpoint_prefix (str, optional): Prefix to be appended to model names while saving to disk. 
                                                 Defaults to "".
        tokenizer_type: (str): Type of tokenizer to be used for training the translation model. 
        max_en_vocab_size (int, optional): Maximum size of the English vocabulary. Defaults to ENGLISH_VOCAB_SIZE.
        max_te_vocab_size (int, optional): Maximum size of the Telugu vocabulary. Defaults to TELUGU_VOCAB_SIZE.
        retrain_tokenizers (bool, optional): Flag to indicate if the tokenizers should be retrained. Defaults to False.
        resume_training (bool, optional): Flag to indicate if the training should be resumed for an existing model.
    """
    if device == DEVICE_GPU:
        # Start recoding the GPU memory usage -- For Debugging purposes only.
        torch.cuda.memory._record_memory_history(max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT)
    # Running this in a try block to capture the memory snapshot even in case of an exception.
    try:
        # Time in seconds elapsed since the beginning of the epoch (Not training epoch but the standard point from which time is calculated).
        # Time at which the model training started.
        start_time = time.time()
        logger.info(f"Loading the following dataset for training: {TRAIN_DATASET_PATH}")
        # Load the train dataset from disk.
        train_dataset: datasets.arrow_dataset.Dataset = load_data_from_disk(dataset_relative_path=TRAIN_DATASET_PATH)
        # Wrap the hugging face dataset in a pytorch Dataset to be able to use with pytorch DataLoader.
        translation_dataset = DatasetWrapper(hf_dataset=train_dataset, dataset_name="TRAIN_DATASET")
        # Load the validation dataset from disk.
        validation_hf_dataset: datasets.arrow_dataset.Dataset = load_data_from_disk(dataset_relative_path=VALIDATION_DATASET_PATH)
        # Wrap the hugging face dataset in a pytorch Dataset to be able to use with pytorch DataLoader.
        validation_dataset = DatasetWrapper(hf_dataset=validation_hf_dataset, dataset_name="VALIDATION_DATASET")
        # Get the tokenizers for the English and Telugu languages.
        english_tokenizer, telugu_tokenizer = get_tokenizers(dataset_relative_path=FULL_EN_TE_DATASET_PATH, 
                                                             tokenizer_type=tokenizer_type,
                                                             retrain_tokenizers=retrain_tokenizers,
                                                             max_en_vocab_size=max_en_vocab_size,
                                                             max_te_vocab_size=max_te_vocab_size)
        pad_token_id = english_tokenizer.get_token_id(PAD_TOKEN)
        # Train the translation model.
        trained_model, model_state = train_model(num_epochs=NUM_EPOCHS, 
                                                 translation_dataset=translation_dataset,
                                                 validation_dataset=validation_dataset,
                                                 english_tokenizer=english_tokenizer, 
                                                 telugu_tokenizer=telugu_tokenizer, 
                                                 pad_token_id=pad_token_id,
                                                 model_name=model_name,
                                                 model_checkpoint_prefix=model_checkpoint_prefix,
                                                 device=device,
                                                 resume_training=resume_training)
        # Save the trained model to disk.
        save_model_to_disk(model=trained_model, model_name=model_name, checkpoint_prefix=model_checkpoint_prefix)
        # Time at which the model training ended.
        end_time = time.time()
        training_time_in_minutes = (end_time - start_time) / 60
        logger.info(f"Model training completed in {training_time_in_minutes} minutes.")
        model_state.set_total_training_time(total_training_time_in_minutes=training_time_in_minutes)
        model_state.log_state()
    except Exception as exp:
        logger.error(f"Model training failed: {exp}")
    finally:
        # Save the memory snapshot to disk.
        try:
            # The last 'MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT' (10000) events will be saved to the snapshot file.
            torch.cuda.memory._dump_snapshot(filename=get_absolute_path(MEMORY_SNAPSHOT_PATH))
        except Exception as e:
            logger.error(f"Failed to capture memory snapshot {e}")
        if device == DEVICE_GPU:
            # Stop recording memory snapshot history.
            torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description="Train the machine translation model.")
    parser.add_argument("--model_checkpoint_prefix", type=str, help="Prefix to be appended to model names while saving to disk")
    parser.add_argument("--model_name", type=str, help="Name to use to save the final model on disk.", default="best_model.pth")
    parser.add_argument("--device", type=str, help="Device to be used for training the model.", default=get_device())
    parser.add_argument("--tokenizer_type", type=str, help="Tokenizer type to be used for model training. Can be 'spacy' or 'bpe'", default="bpe")
    # Please note pre-trained tokenizers already have the vocabulary sizes set and these arguments are only useful if you want
    # to train the tokenizers from scratch. These inputs will be used only if 'retrain_tokenizer' is set to True, will be
    # ignored otherwise.
    parser.add_argument("--max_english_vocab_size", type=int, help="Maximum size of the English vocabulary.", default=ENGLISH_VOCAB_SIZE)
    parser.add_argument("--max_telugu_vocab_size", type=int, help="Maximum size of the Telugu vocabulary.", default=TELUGU_VOCAB_SIZE)
    # This will overwrite the existing tokenizers. This is useful when you want to train the tokenizers.
    # 'Spacy' tokenizers will always be trained during the model training. 
    # 'BPE' tokenizers can be saved and loaded to and from the disk. 
    parser.add_argument("--retrain_tokenizers", type=bool, help="Flag to indicate if the tokenizers should be retrained.", default=False)
    # This will use the 'model_name' and 'model_checkpoint_prefix' to load an existing model from disk and resume training
    # if this flag is set to True.
    parser.add_argument("--resume_training", type=str, help="Flag to indicate if the training should be resumed for an existing model.", default=False)
    args = parser.parse_args()

    logger.info(f"Training the translation model with the following arguments: {args}")
    # Train the translation model.
    train_translation_model(model_name=args.model_name, 
                            model_checkpoint_prefix=args.model_checkpoint_prefix, 
                            device=args.device,
                            tokenizer_type=args.tokenizer_type,
                            max_en_vocab_size=args.max_english_vocab_size,
                            max_te_vocab_size=args.max_telugu_vocab_size,
                            retrain_tokenizers=args.retrain_tokenizers,
                            resume_training=args.resume_training)