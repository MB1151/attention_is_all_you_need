# This file implements the main script to query the trained machine translation model to translate 
# english sentences to telugu.

from model_implementation.data_processing.data_preparation.data_helpers import get_tokenizers
from model_implementation.data_processing.tokenization.bpe_tokenizer import BPETokenizer
from model_implementation.model_building.machine_translation_model import MachineTranslationModel
from model_implementation.model_inference.translator import translate
from model_implementation.utils.constants import ( 
    DEFAULT_BEAM_SIZE, DROPOUT_PROB, D_FEED_FORWARD, D_MODEL, FULL_EN_TE_DATASET_PATH, 
    MAX_INPUT_SEQUENCE_LENGTH, NUM_HEADS, NUM_LAYERS
)
from model_implementation.utils.config import LOG_LEVEL
from model_implementation.utils.helpers import get_device, load_model_from_disk
from model_implementation.utils.logger import get_logger

import argparse
import logging

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


if __name__ == "__main__":
    # Parse the command line arguments.
    parser = argparse.ArgumentParser(description="Query the machine translation model to translate english sentences to telugu.")
    parser.add_argument("--beam_width", type=int, help="Width of the beam to be used in the beam search algorithm.", default=DEFAULT_BEAM_SIZE)
    parser.add_argument("--model_name", type=str, help="Name of the model to load from disk.", default="best_model.pth")
    parser.add_argument("--model_checkpoint_prefix", type=str, help="Prefix to be appended to model names while loading from the disk", default="")
    parser.add_argument("--search_type", type=str, help="Type of search to be used. Can be beam or greedy", default="beam")
    parser.add_argument("--device", type=str, help="Device to be used during model inference.", default=get_device())
    args = parser.parse_args()

    english_tokenizer, telugu_tokenizer = get_tokenizers(dataset_relative_path=FULL_EN_TE_DATASET_PATH, 
                                                         tokenizer_type="bpe", 
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
    
    while True:
        # Note that this processes all the sentences in a single batch. This is not the most efficient way to do it but it 
        # is done to keep the code simple. A much better way would be to process the sentences in multiple batches 
        # depending on the size of the input.
        src_input = input("Enter colon (;) separated source sentences (english) to be translated to Telugu: ")
        src_sentences = src_input.split(";")
        logger.debug(f"Source sentences: {src_sentences}")
        translated_sentences = translate(translation_model=translation_model, 
                                         src_tokenizer=english_tokenizer, 
                                         tgt_tokenizer=telugu_tokenizer, 
                                         src_sentences=src_sentences, 
                                         beam_size=args.beam_width,
                                         search_type=args.search_type,
                                         device=args.device)
        print("Here are the translations: ")
        for english_sentence, translated_telugu_sentence in zip(src_sentences, translated_sentences):
            print(english_sentence, " --> ", translated_telugu_sentence)
        print("\n\n")