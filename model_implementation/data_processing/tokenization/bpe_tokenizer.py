# This file implements the Byte level BPE Tokenizer. It takes in data and trains BPE tokenizers.

from tokenizers import ByteLevelBPETokenizer # type: ignore
from model_implementation.data_processing.tokenization.base_tokenizer import BaseTokenizer
from model_implementation.utils.constants import MAX_VOCAB_SIZE
from model_implementation.utils.helpers import get_absolute_path
from typing import Callable, List, Optional

import datasets


class BPETokenizer(BaseTokenizer):
    """Trains a tokenizer using HuggingFace libraries"""
    def __init__(self, language: str):
        super().__init__(language, "bpe")

    def initialize_tokenizer_and_build_vocab(self, 
                                             data_iterator: datasets.arrow_dataset.Dataset, 
                                             text_extractor: Callable[[dict[str, str], str], str], 
                                             max_vocab_size: Optional[int] = MAX_VOCAB_SIZE):
        self.max_vocab_size = max_vocab_size
        self.tokenizer = self.__train_tokenizer(data_iterator=data_iterator, text_extractor=text_extractor, max_vocab_size=max_vocab_size)

    def tokenize(self, text: str) -> list[str]:
        encoded_text = self.tokenizer.encode(text)
        return encoded_text.tokens

    def encode(self, text: str) -> list[int]:
        encoded_text = self.tokenizer.encode(text)
        return encoded_text.ids

    def decode(self, token_ids: List[int]) -> str:
        """Converts the series of token ids back to the original text.

        Args:
            token_ids (List[int]): A list of token ids corresponding to some text.

        Returns:
            str: The original text corresponding to the token ids.
        """
        return self.tokenizer.decode(token_ids)

    def get_token_id(self, token: str) -> int:
        return self.tokenizer.token_to_id(token)

    def get_vocab_size(self) -> int:
        """Returns the size of the vocabulary i.e., the number of tokens in the vocabulary."""
        return self.tokenizer.get_vocab_size()

    # We need an iterator to train the tokenizer. Using an iterator ensures that not all 
    # the data is loaded into memory at once.
    def __get_data_iterator(self, data_iterator: datasets.arrow_dataset.Dataset, text_extractor: Callable[[dict[str, str], str], str]):
        for data_point in data_iterator:
            yield text_extractor(data_point=data_point, language=self.language) # type: ignore

    def __train_tokenizer(self, data_iterator: datasets.arrow_dataset.Dataset, 
                          text_extractor: Callable[[dict[str, str], str], str], 
                          max_vocab_size: Optional[int]=MAX_VOCAB_SIZE) -> ByteLevelBPETokenizer:
        # Use BPE to train a ByteLevel BPE tokenizer.
        tokenizer = ByteLevelBPETokenizer()
        # train_from_iterator is used so that the entire dataset is not loaded into memory at once.
        tokenizer.train_from_iterator(iterator=self.__get_data_iterator(data_iterator=data_iterator, text_extractor=text_extractor), 
                                      vocab_size= max_vocab_size, 
                                      special_tokens=self.special_tokens)
        return tokenizer

    
    def save_tokenizer_to_disk(self, directory_to_save: str):
        """Saves the trained BPE tokenizer to the disk.

        Args:
            directory_to_save (str): Directory where the tokenizer should be saved.
        """
        absolute_directory_path = get_absolute_path(relative_path=directory_to_save)
        self.tokenizer.save_model(absolute_directory_path)


    def load_trained_tokenizer_from_disk(self, saved_tokenizer_directory: str):
        """Loads the trained BPE tokenizer from the disk.

        Args:
            saved_tokenizer_directory (str): Directory path (relative to the repository root) where the trained 
                                             tokenizer is saved.
        """
        absolute_directory_path = get_absolute_path(relative_path=saved_tokenizer_directory)
        self.tokenizer = ByteLevelBPETokenizer.from_file(vocab_filename=f"{absolute_directory_path}/vocab.json", 
                                                         merges_filename=f"{absolute_directory_path}/merges.txt")
        