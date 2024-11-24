# This file implements Spacy tokenizer. It takes in data and creates the vocabulary using the spacy
# tokenizer models.

from model_implementation.data_processing.tokenization.base_tokenizer import BaseTokenizer
from model_implementation.utils.constants import MAX_VOCAB_SIZE, UNK_TOKEN
from model_implementation.utils.helpers import get_absolute_path
from torchtext.vocab import build_vocab_from_iterator
from typing import Callable, List, Optional

import datasets
import pickle
import spacy


class SpacyTokenizer(BaseTokenizer):
    """Creates a tokenizer that tokenizes the text using the Spacy tokenizer models."""
    def __init__(self, language: str):
        super().__init__(language, "spacy")
    
    def initialize_tokenizer_and_build_vocab(self, 
                                             data_iterator: datasets.arrow_dataset.Dataset, 
                                             text_extractor: Callable[[dict[str, str], str], str], 
                                             max_vocab_size: Optional[int] = MAX_VOCAB_SIZE):
        # Load spacy models for English text tokenization.
        if self.language == "english":
            self.tokenizer = spacy.load("en_core_web_sm").tokenizer          
        elif self.language == "telugu":
            # Load spacy model for Telugu text tokenization.
            self.tokenizer = spacy.blank("te").tokenizer            
        else:
            # Raise an error for unknown language
            pass
        self.max_vocab_size = max_vocab_size
        self.__build_vocab(data_iterator=data_iterator, text_extractor=text_extractor)

    def tokenize(self, text: str) -> list[str]:
        return [token.text for token in self.tokenizer(text)]

    def encode(self, text: str) -> list[int]:
        return self.vocab(self.tokenize(text))

    def decode(self, token_ids: List[int]) -> str:
        """Converts the series of token ids back to the original text.

        Args:
            token_ids (List[int]): A list of token ids corresponding to some text.

        Returns:
            str: The original text corresponding to the token ids.
        """
        token_strings = self.vocab.lookup_tokens(token_ids)
        # Here we are just attaching all the individual token strings with a space in between to form 
        # the original text. This does not give the exact original text but a close approximation. 
        # Using this here for simplicity.
        return " ".join(token_strings)


    def get_token_id(self, token: str) -> int:
        return self.vocab([token])[0]

    def get_vocab_size(self) -> int:
        """Returns the size of the vocabulary i.e., the number of tokens in the vocabulary."""
        return len(self.vocab)

    def __yield_tokens(self, data_iterator: datasets.arrow_dataset.Dataset, text_extractor: Callable[[dict[str, str], str], str]):
        """Returns a generator object that emits tokens for each sentence in the dataset"""
        for data_point in data_iterator:
            yield self.tokenize(text_extractor(dict(data_point), self.language))

    def __build_vocab(self, data_iterator: datasets.arrow_dataset.Dataset, text_extractor: Callable[[dict[str, str], str], str]):
        """Builds the vocabulary for the given dataset"""
        self.vocab = build_vocab_from_iterator(iterator=self.__yield_tokens(data_iterator=data_iterator, text_extractor=text_extractor), 
                                               min_freq=2, 
                                               specials=self.special_tokens, 
                                               special_first=True, 
                                               max_tokens=self.max_vocab_size)
        self.vocab.set_default_index(self.vocab[UNK_TOKEN])
    
    def save_tokenizer_to_disk(self, directory: str):
        """Saves the trained BPE tokenizer to the disk.

        Args:
            directory (str): directory (relative to the repository root) where the vocabulary should be saved.
        """
        absolute_directory_path = get_absolute_path(relative_path=directory)
        with open(f"{absolute_directory_path}/tokenizer.pkl", 'wb') as file_obj:
            pickle.dump(self.vocab, file_obj)

    def load_trained_tokenizer_from_disk(self, directory: str):
        """Loads the trained BPE tokenizer from the disk.

        Args:
            directory (str): directory (relative to the repository root) where the vocabulary is saved.
        """
        absolute_directory_path = get_absolute_path(relative_path=directory)
        with open(f"{absolute_directory_path}/tokenizer.pkl", 'rb') as file_obj:
            en_vocab_loaded = pickle.load(file_obj)
        return en_vocab_loaded