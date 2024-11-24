# Some of the constants here should have been passed as command line arguments when we run the training 
# script. However, I did not feel it was necessary since it is easy to update this file to make the 
# changes take effect and finally, we are going to use some static constants that give the best 
# performance and so I thought it would be better if the best parameters to use are in the constants 
# file for people to see.


# CONSTANTS REALTED TO THE DATASETS.
# Path at which the smaller datasets used by the model are stored.
AI4_BHARAT_DATA_PATH = "Data/AI4Bharat"
# Path at which the data to train the model is stored. Contains 250000 examples.
TRAIN_DATASET_PATH = "Data/AI4Bharat/train_dataset"
# Path at which the a larger dataset to train the model is stored. Contains 500000 examples.
LARGE_TRAIN_DATASET_PATH = "Data/AI4Bharat/large_train_dataset"
# Path at which the data to validate the model is stored.
VALIDATION_DATASET_PATH = "Data/AI4Bharat/validation_dataset"
# Path to a very small dataset that can be used for debuggung purposes.
DEBUG_DATASET_PATH = "Data/AI4Bharat/debug_dataset"
# Path at which the data to test the model is stored. This is used to measure the quality of the model.
TEST_DATASET_PATH = "Data/AI4Bharat/test_dataset"
# Path to the full translation dataset (English - Telugu).
FULL_EN_TE_DATASET_PATH = "Data/AI4Bharat/full_en_te_dataset"


# CONSTANTS RELATED TO THE PRODUCED ARTIFACTS BY THE MODEL.
# Path at which trained bpe english tokenizer is saved.
BPE_ENGLISH_TOKENIZER_SAVE_PATH = "Data/trained_models/tokenizers/bpe/bpe_english_tokenizer"
# Path at which trained bpe telugu tokenizer is saved.
BPE_TELUGU_TOKENIZER_SAVE_PATH = "Data/trained_models/tokenizers/bpe/bpe_telugu_tokenizer"
# Path at which trained spacy english tokenizer is saved.
SPACY_ENGLISH_TOKENIZER_SAVE_PATH = "Data/trained_models/tokenizers/spacy/spacy_english_tokenizer"
# Path at which trained spacy telugu tokenizer is saved.
SPACY_TELUGU_TOKENIZER_SAVE_PATH = "Data/trained_models/tokenizers/spacy/spacy_telugu_tokenizer"
# Path at which the trained model checkpoints are saved during the training process.
MODEL_CHECK_POINT_PATH = "Data/trained_models/translation_models"

# SPECIAL TOKEN CONSTANTS.
# Token to represent the start of a sentence.
START_TOKEN = "<sos>"
# Token to represent the end of a sentence.
END_TOKEN = "<eos>"
# Token to represent padding in the input data.
PAD_TOKEN = "<pad>"
# Token to represent unknown tokens in the input data.
UNK_TOKEN = "<unk>"

# DEVICE RELATED CONSTANTS.
# device to be used for training the model on the GPU.
DEVICE_GPU = "cuda"
# device to be used for training the model on the CPU.
DEVICE_CPU = "cpu"

# HYPER PARAMETERS USED IN THE MODEL.
# Number of tokens in the vocabulary of the tokenizer.
MAX_VOCAB_SIZE = 32000
# Maximum number of tokens in the English vocabulary.
ENGLISH_VOCAB_SIZE = 30000
# Maximum number of tokens in the Telugu vocabulary.
TELUGU_VOCAB_SIZE = 30000
# Default number of workers to be used to load the data from disk. This should
# be set based on the machine being used and a bit of experimentation.
NUM_WORKERS = 8
# Default number of examples to be loaded into a batch. This should be set based
# on the model performace with a bit of experimentation.
BATCH_SIZE = 64
# Default dropout probability to be used in the model.
DROPOUT_PROB = 0.1
# Default probablity to be used for label smoothing ie., the amount of probability to be shared 
# among the tokens excluding the correct token and padding tokens.
SMOOTHING_PROB = 0.1
# Learning rate at the start of the model training. This is adjusted periodically during model training.
INITIAL_LEARNING_RATE = 1
# Hyperparameter to calculate the m1 moment in the optimizer. This roughly corresponds to averaging over the
# last 10 (1/(1-beta_1)) sets of gradients. This comes from 'Gradient Descent with Momentum' algorithm.
BETA_1 = 0.9
# Hyperparameter to calculate the m1 moment in the optimizer. This roughly corresponds to averaging over the
# last 50 (1/(1-beta_2)) sets of gradients. This comes from 'RMS prop' algorithm.
BETA_2 = 0.98
# Small value to avoid division by zero in the optimizer.
EPSILON = 1e-9
# Size of the embeddings and the intermediate vector in the translation model.
D_MODEL = 512
# Size of the hidden layer in the Feed forward neural network in the translation model.
D_FEED_FORWARD = 2048
# Number of heads in each attention layer in the translation model.
NUM_HEADS = 8
# Number of Encoder / Decoder layers in the translation model.
NUM_LAYERS = 6
# Number of steps for which the learning rate increases linearly during training.
NUM_WARMUP_STEPS = 4000
# Maximum number of tokens in a sentence that is every input to the model.
MAX_INPUT_SEQUENCE_LENGTH = 150
# Number of epochs to train the model.
NUM_EPOCHS = 2
# Maximum number of tokens allowed in a translated sentence.
MAX_INFERENCE_SEQ_LEN = 150
# DEFAULT beam size to be used in the Beam Search algorithm.
DEFAULT_BEAM_SIZE = 3
# Number of batches after which the model parameters are updated.
GRADIENT_UPDATE_FREQUENCY = 5


# The maximum number of memory events that can be stored in a snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 10000
# File in which the memory snapshot is saved.
MEMORY_SNAPSHOT_PATH = "Data/trained_models/miscellaneous/memory_snapshot.pickle"