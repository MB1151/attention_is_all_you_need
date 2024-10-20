

from datasets import load_from_disk
from model_implementation.data_processing.tokenization.base_tokenizer import BaseTokenizer
from model_implementation.data_processing.tokenization.bpe_tokenizer import BPETokenizer
from model_implementation.data_processing.tokenization.spacy_tokenizer import SpacyTokenizer
from model_implementation.utils.constants import BPE_ENGLISH_TOKENIZER_SAVE_PATH, BPE_TELUGU_TOKENIZER_SAVE_PATH, ENGLISH_VOCAB_SIZE, TELUGU_VOCAB_SIZE
from model_implementation.utils.helpers import get_absolute_path
from model_implementation.utils.logger import get_logger
from typing import Optional, Tuple

import datasets


logger = get_logger(__name__)


def text_extractor(data_point: dict[str, str], language: str) -> str:
    """Extracts the appropriate text from the example in the dataset based on the language.

    Args:
        data_point (dict[str, str]): A single example from the dataset containing the text in the form of 
                                     a dictionary. The sources sentence is stored in the key 'src' and the 
                                     target sentence is stored in the key 'tgt'.
        language (str): Language of the text to be extracted from the data_point.

    Raises:
        ValueError: Raises an error if the language is not 'english' or 'telugu'.

    Returns:
        str: The text in the data_point.
    """
    if language == "english":
        return data_point["src"]
    elif language == "telugu":
        return data_point["tgt"]
    raise ValueError("Language should be either 'english' or 'telugu'.")


def load_data_from_disk(dataset_relative_path: str) -> datasets.arrow_dataset.Dataset:
    """Loads the dataset from disk.

    Args:
        dataset_relative_path (str): Path to the dataset relative to the repository root.

    Returns:
        datasets.arrow_dataset.Dataset: Returns the dataset loaded from the disk as a hugging face dataset.
    """
    dataset_absolute_path = get_absolute_path(relative_path=dataset_relative_path)
    dataset = load_from_disk(dataset_path=dataset_absolute_path)
    return dataset # type: ignore


def train_tokenizer(dataset: datasets.arrow_dataset.Dataset, tokenizer: BaseTokenizer, max_vocab_size: Optional[int]=32000):
    tokenizer.initialize_tokenizer_and_build_vocab(data_iterator=dataset, 
                                                   text_extractor=text_extractor, 
                                                   max_vocab_size=max_vocab_size)


def get_tokenizers(dataset_relative_path: str, 
                   tokenizer_type: str,
                   retrain_tokenizers: bool=False,
                   max_en_vocab_size: Optional[int]=ENGLISH_VOCAB_SIZE,
                   max_te_vocab_size: Optional[int]=TELUGU_VOCAB_SIZE) -> Tuple[BaseTokenizer, BaseTokenizer]:
    """Creates the tokenziers for the English and Telugu languages by training on the provided dataset.

    Args:
        dataset_relative_path (str): Path to the dataset to train the tokenizers on.
        tokenizer_type (str): Type of tokenizer to be used. It can be either 'spacy' or 'bpe'.
        retrain_tokenizers (bool, optional): Flag to indicate if the tokenizers should be retrained. Defaults to False.
        max_en_vocab_size (int, optional): Maximum size of the English vocabulary to create from the input data corpus.
        max_te_vocab_size (int, optional): Maximum size of the Telugu vocabulary to create from the input data corpus.

    Raises:
        ValueError: Raises an error if the tokenizer_type is not 'spacy' or 'bpe'.

    Returns:
        Tuple[BaseTokenizer, BaseTokenizer]: Returns the tokenizers for the English and Telugu languages.
                                             Returns the English tokenizer first and the Telugu tokenizer second.
    """
    train_dataset = load_data_from_disk(dataset_relative_path=dataset_relative_path)
    if tokenizer_type == "spacy":
        # Always train the spacy tokenizers. This is because I haven't found any easy way to save and load the 
        # 'torchtext.vocab.Vocab' to and from disk.
        english_tokenizer = SpacyTokenizer(language="english")
        train_tokenizer(dataset=train_dataset, tokenizer=english_tokenizer, max_vocab_size=max_en_vocab_size)
        telugu_tokenizer = SpacyTokenizer(language="telugu")
        train_tokenizer(dataset=train_dataset, tokenizer=telugu_tokenizer, max_vocab_size=max_te_vocab_size)
    elif tokenizer_type == "bpe":
        english_tokenizer = BPETokenizer(language="english")
        telugu_tokenizer = BPETokenizer(language="telugu")
        if retrain_tokenizers == False:
            logger.info(f"Loading pre-trained tokenizers from disk")
            # Load the trained tokenizers from the disk if retrain_tokenizers is False.
            english_tokenizer.load_trained_tokenizer_from_disk(saved_tokenizer_directory=BPE_ENGLISH_TOKENIZER_SAVE_PATH)
            telugu_tokenizer.load_trained_tokenizer_from_disk(saved_tokenizer_directory=BPE_TELUGU_TOKENIZER_SAVE_PATH)
        else:
            logger.info(f"Retraining Tokenizers")
            # Train the tokenizers on the dataset if retrain_tokenizers is True.
            train_tokenizer(dataset=train_dataset, tokenizer=english_tokenizer, max_vocab_size=max_en_vocab_size)
            english_tokenizer.save_tokenizer_to_disk(directory_to_save=BPE_ENGLISH_TOKENIZER_SAVE_PATH)
            train_tokenizer(dataset=train_dataset, tokenizer=telugu_tokenizer, max_vocab_size=max_te_vocab_size)
            telugu_tokenizer.save_tokenizer_to_disk(directory_to_save=BPE_TELUGU_TOKENIZER_SAVE_PATH)
    else:
        raise ValueError(f"Tokenizer type {tokenizer_type} is not supported.")
    return english_tokenizer, telugu_tokenizer