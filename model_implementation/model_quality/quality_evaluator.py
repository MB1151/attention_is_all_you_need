# Implements the function to evaluate the quality of the machine translation model on the test dataset.
# The quality is evaluated using BLEU score.

from model_implementation.data_processing.data_preparation.dataset_wrapper import DatasetWrapper
from model_implementation.data_processing.tokenization.base_tokenizer import BaseTokenizer
from model_implementation.model_building.machine_translation_model import MachineTranslationModel
from model_implementation.model_inference.translator import translate
from model_implementation.model_quality.bleu_score import BleuScoreType, calculate_bleu_score
from model_implementation.utils.constants import DEFAULT_BEAM_SIZE
from model_implementation.utils.logger import get_logger

from typing import Generator, List, Tuple


logger = get_logger(__name__)


def yield_batches(dataset: DatasetWrapper, batch_size: int) -> Generator[Tuple[List[str], List[str]], None, None]:
    """Yields batches of the dataset with the given batch size.

    Args:
        dataset (DatasetWrapper): Dataset to be split into batches.
        batch_size (int): Size of the batch to be yielded.

    Yields:
        List[str]: Yields a batch of the dataset.
    """
    for i in range(0, len(dataset), batch_size):
        src_sentences = list(dataset[i:min(i+batch_size, len(dataset))]["src"])  # type: ignore
        tgt_sentences = list(dataset[i:min(i+batch_size, len(dataset))]["tgt"])  # type: ignore
        yield src_sentences, tgt_sentences

def evaluate_quality(machine_translation_model: MachineTranslationModel, 
                     src_tokenizer: BaseTokenizer,
                     tgt_tokenizer: BaseTokenizer,
                     test_dataset: DatasetWrapper, 
                     search_type: str,
                     device: str,
                     beam_width: int=DEFAULT_BEAM_SIZE) -> Tuple[float, float]:
    """Evaluates the quality of the machine translation model on the test dataset.

    Args:
        machine_translation_model (MachineTranslationModel): Trained machine translation model.
        src_tokenizer (BaseTokenizer): Tokenizer for English text.
        tgt_tokenizer (BaseTokenizer): Tokenizer for Telugu text.
        test_dataset (DatasetWrapper): Test dataset to evaluate the model on.
        beam_width (int, optional): Width of the beam to be used in the beam search algorithm. Defaults to 5.
    
    Returns:
        Tuple[float, float]: Returns the BLEU score calculated using teo different methods --> sacrebleu and nltk.
    """
    references = []
    predictions = []
    for src_sentences, tgt_sentences in yield_batches(dataset=test_dataset, batch_size=8):
        # Translate the english sentences to telugu using the model.
        translated_sentences = translate(translation_model=machine_translation_model,
                                         src_tokenizer=src_tokenizer,
                                         tgt_tokenizer=tgt_tokenizer, 
                                         src_sentences=src_sentences,  
                                         beam_size=beam_width,
                                         search_type=search_type,
                                         device=device)
        translated_sentences = tgt_sentences
        references.extend([[tgt_sentence] for tgt_sentence in tgt_sentences])
        predictions.extend(translated_sentences)
    logger.debug(f"Number of references: {len(references)}")
    logger.debug(f"Number of predictions: {len(predictions)}")
    sacre_bleu_score = calculate_bleu_score(references=references, predictions=predictions, score_type=BleuScoreType.SACRE_BLEU)
    nltk_bleu_score = calculate_bleu_score(references=references, predictions=predictions, score_type=BleuScoreType.NLTK_BLEU)
    return sacre_bleu_score, nltk_bleu_score