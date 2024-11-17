# This file implements a pytorch dataloader that reads in raw data (english - telugu sentences), 
# tokenizes the data and creates batches of tokenized data which is used as input to the 
# Translation model. Refer to 'step_4_dataloader_and_with_transformers.ipynb' (link to the notebook)
# notebook to understand dataloaders in detail.

from model_implementation.data_processing.data_preparation.dataset_wrapper import DatasetWrapper
from model_implementation.data_processing.tokenization.base_tokenizer import BaseTokenizer
from model_implementation.utils.constants import BATCH_SIZE, END_TOKEN, NUM_WORKERS, PAD_TOKEN, START_TOKEN
from model_implementation.utils.logger import get_logger
from torch import Tensor
from torch.utils.data import DataLoader, Sampler
from typing import Dict, List, Iterator, Tuple

import random
import torch


logger = get_logger(__name__)


# DataLoader expects the sampler to provide the indices of the data points from the Dataset. These indices are used 
# to construct the batches internally that are later passed to collate_fn. We are creating a sampler which groups 
# sentences of similar lengths into batches. We still need to be aware that the grouping is not perfect and it is 
# done here based on the number of words separated by 'spaces' in the sentence and not the actual number of tokens
# we obtain while using the tokenization algorithm. The actual number of tokens could be very different depending on 
# the tokenization algorithm which splits sentences based on a number of factors.
class __LengthAwareSampler(Sampler):
    def __init__(self, dataset: DatasetWrapper, batch_size: int):
        # dataset is the Dataset wrapper we created on top of HuggingFace dataset.
        self.dataset = dataset
        self.batch_size = batch_size
        self.sorted_indices = self.extract_lengths()
    
    # We don't want the entire dataset to be loaded into the memory at once. So, we first iterate over the entire 
    # dataset, extract the lengths of the sentences and sort the indices of the sentences according to the sentence 
    # lengths. This is to ensure that sentences of similar lengths are grouped together in a batch to minimize the 
    # overall padding necessary. When we iterate over the dataset, we only load the necessary data from the dataset 
    # into memory and not the entire dataset at once --> This loading logic could be a little different based on the 
    # hugging face implementations. Please look into the hugging face documentation for more details.
    def extract_lengths(self) -> list[int]:
        """Sorts the indices of the dataset based on the lengths of the sentences in the dataset.

        Returns:
            list[int]: Indices of the dataset sorted in ascending order (small to big) based on the lengths of 
                       the sentences in the dataset.
        """
        # Note that the lengths are calculated based on the number of words separated by space in the sentence and 
        # not the number of tokens.
        self.lengths = [len(data_point["src"].split(" ")) + len(data_point["tgt"].split(" ")) for data_point in self.dataset]
        # Create an indices list in the first step i.e., [0, 1, 2, ..., 299] --> For debug_dataset with 200 examples.
        # Sorts the indices list based on the calculated in the previous step i.e., after sorting
        # value at 0th index is the example for which the len(src_sentence) + len(tgt_sentence) is minimum and
        # value at 199th index is the example for which the len(src_sentence) + len(tgt_sentence) is maximum.
        return sorted(range(len(self.dataset)), key=lambda index: self.lengths[index])

    # The __iter__ function is called once per epoch. The returned iterator is iterated on to get the list of indices 
    # for the data points in a batch.
    def __iter__(self) -> Iterator[int]:
        """Provides an iterator that yields the indices of the dataset in the order of the sentence lengths.

        Returns:
            Iterator[int]: An iterator that contains the indices of the dataset in the order of the sentence lengths.
        """
        # Create the batches of indices based on the sentence lengths. 
        # batches look like: [[0, 5, 90], [23, 4, 5], ...] if batch_size is 3.
        # [0, 5, 90] is a batch corresponding to the sentences at indices 0, 5 and 90 in the original dataset.
        # [23, 4, 5] is a batch corresponding to the sentences at indices 23, 4 and 5 in the original dataset.
        batches = [self.sorted_indices[index: index + self.batch_size] for index in range(0, len(self.dataset), self.batch_size)]
        # Shuffle the batches to ensure that the order of batches is different in every epoch. We want the model to 
        # see the data in different order in every epoch. So, we shuffle the order of the batches within the dataset.
        random.shuffle(batches)
        # Flatten the list of batches to get an iterable of indices. At the end, the dataloader expects an iterable of
        # indices to get the data points from the dataset. So, we convert the list of batches back to an iterable of 
        # indices.
        return iter([index for batch in batches for index in batch])
    


def __length_aware_collate_fn(batch: List[Dict], 
                              english_tokenizer: BaseTokenizer, 
                              telugu_tokenizer: BaseTokenizer, 
                              sos_id: int, 
                              eos_id: int, 
                              pad_id: int) -> Tuple[Tensor, Tensor]:
    """Converts the raw data in the batch into the format required by the MachineTranslationTransformer model. It encodes the
       sentences into token ids, adds start, end and padding tokens and converts the raw batch into batched tensors to be 
       used by the model.

    Args:
        batch (List[Dict]): Holds the raw data points (the actual english (src) and telugu (tgt) sentences batched) from the dataset.
        english_tokenizer (BaseTokenizer): Tokenizer to tokenize and encode the english sentences into corresponding token ids.
        telugu_tokenizer (BaseTokenizer): Tokenizer to tokenize and encode the english sentences into corresponding token ids.
        sos_id (int): start of sentence token id. Usually, this is 0.
        eos_id (int): end of sentence token id. Usually, this is 1.
        pad_id (int): padding token id. Usually, this is 2.

    Returns:
        Tuple[Tensor, Tensor]: Returns the encoded source and target tensors in the batch which can be used by the transformer 
                               model as input.
                               shape of Tensor: [BATCH_SIZE, MAX_SENTENCE_LENGTH]
    """
    # Holds all the encoded src sentences (english sentences) from the batch. encoded sentence means sentence divided 
    # into tokens and tokens converted into their integer ids.
    # [[0, 223, 4345, 545, 1], [0, 23, 234, 67, 1]] is an example for the processed_src_sentences variable where
    # [0, 223, 4345, 545, 1] represents an encoded sentence from the batch and 0 at the start is <sos> and 1 at the 
    # end is <eos>. 
    processed_src_sentences = []
    # Holds all the encoded tgt sentences (telugu sentences) from the batch.
    processed_tgt_sentences = []
    for data_point in batch:
        # src is english sentence.
        src_sentence = data_point["src"]
        # tgt is telugu sentence.
        tgt_sentence = data_point["tgt"]
        # start of sentence id to append at the start of every sentence.
        sos_tensor = torch.tensor([sos_id], dtype=torch.int64)
        # end of sentence id to append at the end of every sentence.
        eos_tensor = torch.tensor([eos_id], dtype=torch.int64)
        # It is important to set the dtype to 'torch.int64' because we map token_ids to their embeddings in the transformer model.
        # '<sos>' and '<eos>' tokens are not added to the src sentences. They are only added to the target sentences.
        encoded_src_sentence = torch.tensor(english_tokenizer.encode(src_sentence), dtype=torch.int64)
        # prepares the tensor in the format 'token_id(<sos>) token_id1 token_id2 ... last_token_id token_id(<eos>)'. 
        encoded_tgt_sentence = torch.cat([sos_tensor, torch.tensor(telugu_tokenizer.encode(tgt_sentence), dtype=torch.int64), eos_tensor], dim=0)
        processed_src_sentences.append(encoded_src_sentence)
        processed_tgt_sentences.append(encoded_tgt_sentence)
    # Finds the maximum length of the src_sequences in the batch so that src sequences are padded to get all the sequences
    # to the same length i.e., max_src_seq_len.
    max_src_seq_len = max(src_ids.size(0) for src_ids in processed_src_sentences)
    # Finds the maximum length of the tgt_sequences in the batch so that src sequences are padded to get all the sequences
    # to the same length i.e., max_tgt_seq_len.
    max_tgt_seq_len = max(tgt_ids.size(0) for tgt_ids in processed_tgt_sentences)
    # We pad the sentences with pad token so that every sentence in the batch is of same length. Also, notice 
    # that the pad token is appended after (not before) the <eos> token is appended to every sentence.
    src_ids = [torch.nn.functional.pad(input=src_ids, pad=(0, max_src_seq_len - src_ids.size(0)), mode="constant", value=pad_id) for src_ids in processed_src_sentences]
    tgt_ids = [torch.nn.functional.pad(input=tgt_ids, pad=(0, max_tgt_seq_len - tgt_ids.size(0)), mode="constant", value=pad_id) for tgt_ids in processed_tgt_sentences]
    # stack the src tensors along dimension 0. This then becomes a 2D tensor of shape (BATCH_SIZE, MAX_SENTENCE_LENGTH).
    src = torch.stack(tensors=src_ids, dim=0)
    tgt = torch.stack(tensors=tgt_ids, dim=0)
    return (src, tgt)



def create_data_loader(dataset: DatasetWrapper, 
                       english_tokenizer: BaseTokenizer, 
                       telugu_tokenizer: BaseTokenizer, 
                       num_workers: int=NUM_WORKERS, 
                       batch_size: int=BATCH_SIZE) -> DataLoader:
    """Tokenizes the translation dataset (English-Telugu), creates batches out of the tokenized data and wraps
       it in pytorch dataloaders to be used by the Machine Translation model.

    Args:
        dataset (DatasetWrapper): torch dataset to be used to create the dataloader for.
        english_tokenizer (BaseTokenizer): English language tokenizer (spacy or Byte level BPE).
        telugu_tokenizer (BaseTokenizer): Telugu language tokenizer (spacy or Byte level BPE).
        num_workers (int, optional): Number of workers to be used to load the data parallely. Defaults to NUM_WORKERS.
        batch_size (int, optional): Size of the batch. Defaults to BATCH_SIZE.

    Returns:
        DataLoader: Returns a pytorch dataloader which can be iterated upon to get the data for the model to train on.
    """
    logger.info(f"Creating DataLoader object for the following dataset: {dataset.dataset_name}")
    length_aware_sampler = __LengthAwareSampler(dataset=dataset, batch_size=batch_size)
    # We make sure that token ids for start, end and pad tokens are same in both the tokenizers. So, 
    # here we use english_tokenizer to find the ids which can be replaced by telugu_tokenizer if you 
    # want.
    start_token_id = english_tokenizer.get_token_id(token=START_TOKEN)
    end_token_id = english_tokenizer.get_token_id(token=END_TOKEN)
    pad_token_id = english_tokenizer.get_token_id(token=PAD_TOKEN)
    def collate_fn(batch):
        """The collate_fn is called by the DataLoader with just the batch of data points from the dataset. So, we wrap the 
           length_aware_collate_fn function with the required parameters to create a collate_fn that can be used by the 
           DataLoader.

        Args:
            batch (_type_): Batch of raw data points from the dataset.

        Returns:
            _type_: Returns the batch of data points in the format required by the MachineTranslationTransformer model.
        """
        return __length_aware_collate_fn(batch=batch, 
                                         english_tokenizer=english_tokenizer, 
                                         telugu_tokenizer=telugu_tokenizer, 
                                         sos_id=start_token_id, 
                                         eos_id=end_token_id, 
                                         pad_id=pad_token_id)
    return DataLoader(dataset=dataset, 
                      batch_size=batch_size, 
                      sampler=length_aware_sampler, 
                      num_workers=num_workers, 
                      collate_fn=collate_fn)
