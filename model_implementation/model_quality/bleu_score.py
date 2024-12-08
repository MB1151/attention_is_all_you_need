# This file implements multiple methods to calculate bleu score giving reference and predicted sentences.
# references -- True sentences which are known to be correct and against which the predicted sentences are compared.
# predictions -- Sentences predicted by the model.

import evaluate

from datasets import load_metric
from enum import Enum
from nltk.translate.bleu_score import corpus_bleu
from typing import List


class BleuScoreType(Enum):
    SACRE_BLEU = 1
    NLTK_BLEU = 2


def compute_sacre_bleu_score(predictions: List[str], references: List[List[str]]) -> float:
    """Calculates the bleu score using sacrebleu.

    Args:
        predictions (List[str]): Sentences predicted by the model.
        references (List[str]): Each internal list corresponds to a list of true translations for the
                                corresponding prediction in the predictions. 

    Returns:
        float: corpus bleu score for the input. Range: [0, 100]
    """
    # Load the BLEU metric from the datasets library.
    bleu_metric = evaluate.load("sacrebleu")
    bleu_score = bleu_metric.compute(predictions=predictions, references=references)   # type: ignore
    return bleu_score["score"]  # type: ignore


def compute_nltk_bleu_score(predictions: List[str], references: List[List[str]]) -> float:
    """Calculates the bleu score using nltk.

    Args:
        predictions (List[str]): Sentences predicted by the model.
        references (List[List[str]]): Each internal list corresponds to a list of true translations for the
                                      corresponding prediction in the predictions. 

    Returns:
        float: corpus bleu score for the input. Range: [0, 100]
    """
    return corpus_bleu(list_of_references=references, hypotheses=predictions) * 100    # type: ignore


def calculate_bleu_score(predictions: List[str], references: List[List[str]], score_type: BleuScoreType=BleuScoreType.NLTK_BLEU) -> float:
    """Calculate the BLEU score for the given predictions and references.

    Args:
        predictions (List[str]): List of predicted translations.
        references (List[List[str]]): Each internal list corresponds to a list of true translations for the
                                      corresponding prediction in the predictions. 

    Returns:
        float: Returns the BLEU score for the given predictions and references. Range: [0, 100]
    """
    if score_type == BleuScoreType.SACRE_BLEU:
        return compute_sacre_bleu_score(predictions=predictions, references=references)
    elif score_type == BleuScoreType.NLTK_BLEU:
        return compute_nltk_bleu_score(predictions=predictions, references=references)
    else:
        raise ValueError(f"Invalid BLEU score type: {score_type}")