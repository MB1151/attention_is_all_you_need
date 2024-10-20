# This file implements the classes that hold the state of the training and model during model training.

from dataclasses import dataclass
from model_implementation.utils.logger import get_logger
from typing import Optional

logger = get_logger(__name__)

@dataclass
class TrainState:
    """Holds the state of the training about a particular epoch."""
    epoch_num: int
    training_time_in_minutes: float
    training_loss: float
    num_tokens_processed: int
    model_checkpoint_path: str
    num_batches_skipped: Optional[int] = 0
    validation_loss: Optional[float] = None
    epoch_start_learning_rate: Optional[float] = None
    epoch_end_learning_rate: Optional[float] = None

    def log_state(self):
        """Logs the state of the training at a particular point in time."""
        logger.info(f"Epoch: {self.epoch_num}")
        logger.info(f"Training Time (in minutes): {self.training_time_in_minutes}")
        logger.info(f"Epoch start learning Rate: {self.epoch_start_learning_rate}")
        logger.info(f"Epoch end learning Rate: {self.epoch_end_learning_rate}")
        logger.info(f"Training Loss: {self.training_loss}")
        logger.info(f"Num Tokens Processed: {self.num_tokens_processed}")
        logger.info(f"Model Checkpoint Path: {self.model_checkpoint_path}")
        logger.info(f"Number of batches skipped: {self.num_batches_skipped}")
        if self.validation_loss:
            logger.info(f"Validation Loss: {self.validation_loss}")
        print("-" * 150)


class ModelState:
    """Holds the complete state of the training for the model."""
    def __init__(self, device: str):
        # Total training time in minutes. This includes the time taken for loading the data, creating the tokenizers,
        # and training all the epochs. This is only populated correctly at the end of the training.
        self.total_training_time_in_minutes = 0
        # Device (cpu or cuda) on which the model is trained.
        self.device = device
        self.state = []

    def append_state(self, train_state: TrainState):
        """Appends the training state to the list of states."""
        self.state.append(train_state)
    
    def set_total_training_time(self, total_training_time_in_minutes: float):
        """Sets the total training time in minutes."""
        self.total_training_time_in_minutes = total_training_time_in_minutes

    def get_total_training_time(self):
        """Returns the total training time in minutes."""
        return self.total_training_time_in_minutes

    def log_state(self):
        """Logs the state of the model during training."""
        logger.info(f"Device used for training: {self.device}")
        for state in self.state:
            state.log_state()
        logger.info(f"Total Training Time (in minutes): {self.total_training_time_in_minutes}")

    def plot_loss_variation(self):
        pass