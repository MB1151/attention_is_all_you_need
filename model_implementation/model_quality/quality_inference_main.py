# This file implements the main script to query the trained machine translation model to translate
# english sentences to telugu on the testset and measure the quality of the model using BLEU score.

from model_implementation.data_processing.data_preparation.data_helpers import get_tokenizers, load_data_from_disk
from model_implementation.data_processing.data_preparation.dataset_wrapper import DatasetWrapper
from model_implementation.model_building.machine_translation_model import MachineTranslationModel
from model_implementation.model_quality.quality_evaluator import evaluate_quality
from model_implementation.utils.config import LOG_LEVEL
from model_implementation.utils.constants import (
    D_MODEL, D_FEED_FORWARD, DEFAULT_BEAM_SIZE, DROPOUT_PROB, FULL_EN_TE_DATASET_PATH, NUM_HEADS, 
    NUM_LAYERS, MAX_INPUT_SEQUENCE_LENGTH, TEST_DATASET_PATH
)
from model_implementation.utils.helpers import get_device, load_model_from_disk
from model_implementation.utils.logger import get_logger

import argparse
import datasets
import logging


# Set the logging level for the entire quality inference run. This get propogated to all the modules.
logging.basicConfig(level=LOG_LEVEL)
logger = get_logger(name=__name__)


def load_translation_model_from_disk(model_name: str, 
                                     src_vocab_size: int, 
                                     tgt_vocab_size: int, 
                                     checkpoint_prefix: str="") -> MachineTranslationModel:
    """Loads the trained translation model from disk.

    Args:
        model_name (str): Name of the model to load from disk.

    Returns:
        MachineTranslationModel: Returns the trained machine translation model.
    """
    translation_model = MachineTranslationModel(d_model=D_MODEL, 
                                                d_feed_forward=D_FEED_FORWARD,
                                                dropout_prob=DROPOUT_PROB, 
                                                num_heads=NUM_HEADS, 
                                                src_vocab_size=src_vocab_size, 
                                                tgt_vocab_size=tgt_vocab_size, 
                                                num_layers=NUM_LAYERS, 
                                                max_seq_len=MAX_INPUT_SEQUENCE_LENGTH)
    load_model_from_disk(model=translation_model, model_name=model_name, checkpoint_prefix=checkpoint_prefix)
    return translation_model


def load_test_set() -> DatasetWrapper:
    # Load the test dataset from disk.
    test_dataset: datasets.arrow_dataset.Dataset = load_data_from_disk(dataset_relative_path=TEST_DATASET_PATH)
    # Wrap the hugging face dataset in a pytorch Dataset to be able to use with pytorch DataLoader.
    return DatasetWrapper(hf_dataset=test_dataset, dataset_name="TEST_DATASET")



if __name__ == "__main__":
    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description="Query the machine translation model to translate english sentences to telugu.")
    parser.add_argument("--beam_width", type=int, help="Width of the beam to be used in the beam search algorithm.", default=DEFAULT_BEAM_SIZE)
    parser.add_argument("--model_name", type=str, help="Name of the model to load from disk.", default="best_model.pth")
    parser.add_argument("--model_checkpoint_prefix", type=str, help="Prefix to be appended to model names while loading from the disk", default="")
    parser.add_argument("--tokenizer_type", type=str, help="Type of tokenizer to be used. Can be 'spacy' or 'bpe'", default="bpe")
    parser.add_argument("--search_type", type=str, help="Type of search to be used. Can be beam or greedy", default="beam")
    parser.add_argument("--device", type=str, help="Device to be used during model inference.", default=get_device())
    parser.add_argument("--use_saved_predictions", type=bool, help="Flag to use the saved predictions instead of re-evaluating.", default=False)
    args = parser.parse_args()

    english_tokenizer, telugu_tokenizer = get_tokenizers(dataset_relative_path=FULL_EN_TE_DATASET_PATH, 
                                                         tokenizer_type=args.tokenizer_type, 
                                                         retrain_tokenizers=False)
    translation_model = load_translation_model_from_disk(model_name=args.model_name, 
                                                         src_vocab_size=english_tokenizer.get_vocab_size(), 
                                                         tgt_vocab_size=telugu_tokenizer.get_vocab_size(),
                                                         checkpoint_prefix=args.model_checkpoint_prefix)
    # Move the model to the device so that the parameters of the model are stored on gpu and the computations (inference) 
    # are done on gpu.
    translation_model.to(args.device)
    # Set the model to evaluation mode. This is important as the model has different behavior during training and evaluation.
    # For example, dropout is applied only during training and not during evaluation.
    translation_model.eval()

    test_dataset: DatasetWrapper = load_test_set()
    sacrebleu_score, nltk_bleu_score = evaluate_quality(machine_translation_model=translation_model,
                                                        src_tokenizer=english_tokenizer,
                                                        tgt_tokenizer=telugu_tokenizer,
                                                        test_dataset=test_dataset,
                                                        search_type=args.search_type,
                                                        device=args.device,
                                                        beam_width=args.beam_width,
                                                        use_saved_predictions=args.use_saved_predictions)
    logger.info(f"BLEU Score using sacrebleu: {sacrebleu_score}")
    logger.info(f"BLEU Score using nltk: {nltk_bleu_score}")