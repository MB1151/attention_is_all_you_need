# Implements the functionality to evaluate the quality of the machine translation model on the test dataset.
# The quality is evaluated using BLEU score.

from model_implementation.data_processing.data_preparation.dataset_wrapper import DatasetWrapper
from model_implementation.data_processing.tokenization.base_tokenizer import BaseTokenizer
from model_implementation.model_building.machine_translation_model import MachineTranslationModel
from model_implementation.model_inference.translator import translate
from model_implementation.model_quality.bleu_score import BleuScoreType, calculate_bleu_score
from model_implementation.utils.constants import (
    BLEU_BATCH_SIZE, BLEU_PREDICTIONS_FILE_PATH, BELU_REFERENCES_FILE_PATH, DEFAULT_BEAM_SIZE
)
from model_implementation.utils.helpers import save_lines_to_file, load_lines_from_file
from model_implementation.utils.logger import get_logger

from tqdm import tqdm
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


def load_references_and_predictions() -> Tuple[List[List[str]], List[str]]:
    """Loads the references and predictions from the disk.

    Returns:
        Tuple[List[List[str]], List[str]]: Returns the references and predictions as a tuple.
    """
    loaded_references: List[str] = load_lines_from_file(relative_filepath=BELU_REFERENCES_FILE_PATH)
    # Each source sentence can have multiple target sentences as references. Here, we only have one
    # reference for each source sentence.
    references: List[List[str]] = [[reference] for reference in loaded_references]
    loaded_predictions: List[str] = load_lines_from_file(relative_filepath=BLEU_PREDICTIONS_FILE_PATH)
    return references, loaded_predictions


def evaluate_quality(machine_translation_model: MachineTranslationModel, 
                     src_tokenizer: BaseTokenizer,
                     tgt_tokenizer: BaseTokenizer,
                     test_dataset: DatasetWrapper,
                     device: str, 
                     search_type: str="beam",
                     beam_width: int=DEFAULT_BEAM_SIZE,
                     use_saved_predictions: bool=False) -> Tuple[float, float]:
    """Evaluates the quality of the machine translation model on the test dataset.

    Args:
        machine_translation_model (MachineTranslationModel): Trained machine translation model.
        src_tokenizer (BaseTokenizer): Tokenizer for English text.
        tgt_tokenizer (BaseTokenizer): Tokenizer for Telugu text.
        test_dataset (DatasetWrapper): Test dataset to evaluate the model on.
        device (str): Device to be used during model inference. Can be 'cpu' or 'cuda'.
        search_type (str): Type of search to be used. Can be 'beam' or 'greedy'. Defaults to 'beam'.
        beam_width (int, optional): Width of the beam to be used in the beam search algorithm. Defaults to 3.
        use_saved_predictions (bool, optional): Whether to use the saved predictions from the disk. Defaults to False.
        
    Returns:
        Tuple[float, float]: Returns the BLEU score calculated using two different methods --> sacrebleu and nltk.
    """
    references: List[List[str]] = []
    predictions: List[str] = []
    if use_saved_predictions == False:
        # Predict the translations for the test dataset.
        for batch_num, (src_sentences, tgt_sentences) in enumerate(tqdm(yield_batches(dataset=test_dataset, batch_size=BLEU_BATCH_SIZE))):
            logger.info(f"Translating batch number: {batch_num + 1}")
            # Translate the english sentences to telugu using the model.
            # translated_sentences: List[str] = ["CHECK"] * BLEU_BATCH_SIZE
            translated_sentences: List[str] = translate(translation_model=machine_translation_model,
                                            src_tokenizer=src_tokenizer,
                                            tgt_tokenizer=tgt_tokenizer, 
                                            src_sentences=src_sentences,  
                                            beam_size=beam_width,
                                            search_type=search_type,
                                            device=device)
            # Each source sentence can have multiple target sentences as references. Here, we only have one 
            # reference for each source sentence.
            references.extend([[tgt_sentence] for tgt_sentence in tgt_sentences])
            predictions.extend(translated_sentences)
    else:
        # Load the references and predictions from the disk.
        references, predictions = load_references_and_predictions()
    
    logger.debug(f"Number of references: {len(references)}")
    logger.debug(f"Number of predictions: {len(predictions)}")
    
    # Save the references and predictions to disk.
    save_lines_to_file(lines=[reference[0] for reference in references], relative_filepath=BELU_REFERENCES_FILE_PATH)
    save_lines_to_file(lines=predictions, relative_filepath=BLEU_PREDICTIONS_FILE_PATH)

    logger.info("Calculating BLEU score using references and predictions")
    sacre_bleu_score = calculate_bleu_score(references=references, predictions=predictions, score_type=BleuScoreType.SACRE_BLEU)
    nltk_bleu_score = calculate_bleu_score(references=references, predictions=predictions, score_type=BleuScoreType.NLTK_BLEU)
    return sacre_bleu_score, nltk_bleu_score