# This file implements a base class that is to be inherited by all the tokenizer classes.
# This class is created so that different tokenizers can be integrated in a homogenous way into
# the data preparation module.

from abc import ABC, abstractmethod
from model_implementation.utils.constants import START_TOKEN, END_TOKEN, PAD_TOKEN, UNK_TOKEN, MAX_VOCAB_SIZE
from typing import Callable, Optional, List

import datasets


# This is a base class that will be inherited by the actual tokenizer classes.
class BaseTokenizer(ABC):
    """A class created to hold different kinds of tokenizers and handle the token encoding in a common way.
       Here, we only use SpacyTokenizer and HuggingFaceTokenizer."""
    def __init__(self, language: str, tokenizer_type: str):
        self.language = language
        self.tokenizer_type = tokenizer_type
        self.special_tokens = [START_TOKEN, END_TOKEN, PAD_TOKEN, UNK_TOKEN]

    # Abstract methods need to be overridden by the child class. It raises TypeError if not overridden.
    @abstractmethod
    def initialize_tokenizer_and_build_vocab(self, 
                                             data_iterator: datasets.arrow_dataset.Dataset, 
                                             text_extractor: Callable[[dict[str, str], str], str], 
                                             max_vocab_size: Optional[int] = MAX_VOCAB_SIZE):
        """Initializes the tokenizers and builds the vocabulary for the given dataset.

        Args:
            data_iterator (datasets.arrow_dataset.Dataset): An iterator that gives input sentences (text) when iterated upon.
            text_extractor (Callable[[dict[str, str], str], str]): A function that extracts the appropriate text from the input 
                dataset. This parameter is added to make the tokenizer class independent of the input dataset format. If not 
                provided as an argument, we will have to extract the text from the dataset within the 'CustomTokenizer' class 
                which makes it dependent on the dataset format. 
            max_vocab_size (int, optional): Maximum size of the vocabulary to create from the input data corpus. Defaults 
                                            to MAX_VOCAB_SIZE (40000).
        """
        pass

    # Abstract methods need to be overridden by the child class. It raises TypeError if not overridden.
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Returns the individual tokens (possibly readable text) for the given text."""
        pass

    # Abstract methods need to be overridden by the child class. It raises TypeError if not overridden.
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Returns the encoded token ids for the given text."""
        pass

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Converts the series of token ids back to the original text.

        Args:
            token_ids (List[int]): A list of token ids corresponding to some text.

        Returns:
            str: The original text corresponding to the token ids.
        """
        pass

    # Abstract methods need to be overridden by the child class. It raises TypeError if not overridden.
    @abstractmethod
    def get_token_id(self, token: str) -> int:
        """Returns the token id for the given token. If the token is not present in the vocabulary, it returns None."""
        pass

    # Abstract methods need to be overridden by the child class. It raises TypeError if not overridden.
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Returns the size of the vocabulary i.e., the number of tokens in the vocabulary."""
        pass

    @abstractmethod
    def save_tokenizer_to_disk(self, directory: str):
        """Saves the trained BPE tokenizer to the disk.

        Args:
            directory (str): Directory (relative to the repository root) where the tokenizer should be saved.
        """
        pass

    @abstractmethod
    def load_trained_tokenizer_from_disk(self, directory: str):
        """Loads the trained BPE tokenizer from the disk.

        Args:
            directory (str): Directory (relative to the repository root) where the trained tokenizer is saved.
        """
        pass