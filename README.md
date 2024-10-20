# Implementation Of Attention Is All You Need Paper

This repository contains Pytorch implementation of the **English - Telugu** translation model based on the first transformer architecture released in the *Attention Is All You Need* paper (<u>[Link to the paper](https://arxiv.org/abs/1706.03762)</u>).

This repository contains two parallel sets of implementations for the same model:

- Step by Step implementation in the Jupiter notebooks where each line of code is explained in detail and executed.
- Modularized implementation of the same model in python scripts which is eventually used for training and inference.

## Audience For This Repository

This repository is intended for anyone looking to learn about transformers in detail. It contains all the resources necessary to understand the basics of Transformer and implement the English-Telugu translation model.

## Table Of Contents

- [Getting Started](#getting-started)
- [Useful Resources](#useful-resources)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Hardware](#hardware)


## Getting Started

### Learning Transformers (Recommendation)

- In Step 1, start with [Core Concepts](#core-concepts) to learn the basics of neural networks, RNNs, transformers and a overview of the translation model architecture in the *Attention Is All You Need* paper. Ignore any additional dependencies for now.
- In step 2, learn how [Data Preparation](#data-preparation) works at a high level for the transformer based translation model. 
- Step 3 is to understand the [Dependent Concepts](#dependent-concepts) that are required to implement any large Deep Learning models in general.
- Step 4 is to understand the ideas needed to train the model once the core model implementation is in place. Understand the concepts mentioned in [Model Training](#model-training) section.
- In Step 5, understand how inference works in translation model from the [Model Inference](#model-inference) section.
- In Step 6, deep dive into the [Implementation](#implementation) details which will need considerable knowledge of the pytorch framework.
    * Start with deep diving into the `building_transformers_step_by_step/` and understand the implementation. These notebooks point (when needed) to my other repository (understanding_pytorch) which explains how several pytorch functions work.


## Useful Resources

- [Core Concepts](#core-concepts) &rarr; Resources necessary to understand transformers and translation model architecture.
- [Dependent Concepts](#dependent-concepts) &rarr; Resources necessary to implement any large Deep Learning model in general.
- [Data Preparation](#data-preparation) &rarr; Resources necessary to understand the data preparation stage.
- [Model Training](#model-training) &rarr; Resources necessary to understand the translation model training process.
- [Model Inference](#model-inference) &rarr; Resources necessary to understand the translation model inference process.
- [Implementation](#implementation) &rarr; Resources necessary to understand the details required to implement the model.

### Core Concepts

Understanding the core concepts will give an good enough understanding of the model architecture itself. However, there are several dependent concepts which need to be understood inorder to implement the model.

#### <u>Basics of Neural Networks</u>

The transformers are built on top of the Neural Networks and hence it is necessary to be familiar with Neural Networks and Back Propogation.

- <u>[Video](https://www.youtube.com/watch?v=QDX-1M5Nj7s&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=1)</u> giving the summary of Neural Networks &rarr; *By Alexander Amini*
- <u>[Video](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)</u> with visual explanation of Neural Networks &rarr; *By 3Blue1Brown channel*
- <u>[Video](https://www.youtube.com/watch?v=Ilg3gGewQ5U)</u> giving an intuitive explanation of Back Propagation without the Math &rarr; *By 3Blue1Brown channel*
- <u>[Video](https://www.youtube.com/watch?v=tIeHLnjs5U8)</u> explaining the mathematics behing Back Propogation &rarr; *By 3Blue1Brown channel*
- <u>[Blog](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)</u> that runs Back Propogation on an example manually without code &rarr; *By Matt Mazur* 


#### <u>Word Embeddings</u>

The input is divided into tokens which are represented as vectors in the model. Traditionally, these vectors were called word embeddings.

- <u>[Video](https://www.youtube.com/watch?v=jQTuRnjJzBU&list=PLhWB2ZsrULv-wEM8JDKA1zk8_2Lc88I-s&index=1)</u> explaining the basics of Word Embeddings &rarr; *By Andrew NG*
- <u>[Blog](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/)</u> discussing word embeddings in detail and how word embeddings are created traditionally &rarr; *By Adrian Colyer*
- <u>[Video](https://www.youtube.com/watch?v=D-ekE-Wlcds)</u> that discusses how to train models to generate word embeddings and visualize the embeddings &rarr; *By late Xin Rong*


#### <u>RNNs and Transformers</u>

Transformers are an improvement on top of RNNs and Attention is the core of Transformers. The impact of transformers becomes clear if the issues with RNN are understood.

- <u>[Video](https://www.youtube.com/watch?v=ySEx_Bqxvvo&t=827s)</u> giving a quick summary of RNNs and Self Attention &rarr; *By Ava Amini*
- <u>[Blog](https://jalammar.github.io/illustrated-transformer/)</u> explaining the transformer architecture from *Attention Is All You Need* paper. One of the best resources out there to understand the basics of Transformers &rarr; *By Jay Alammar*
- <u>[Video](https://www.youtube.com/watch?v=wjZofJX0v4M)</u> giving a quick introduction to transformers &rarr; *By 3Blue1Brown channel*
- <u>[Video](https://www.youtube.com/watch?v=eMlx5fFNoYc)</u> explaning Attention mechanism in transformers &rarr; *By 3Blue1Brown channel*
    * Videos by 3Blue1Brown are extremely good and intuitive but they are fast paced. So, it would help to have some context on Transformers before watching these videos.


#### <u>Positional Encoding</u>

These resources will provide more information on top of what is already provided in the Jay Alammar's blog about Positional Encoding. So, suggest understanding the Positional Encoding part from the Jay Alammar's blog as a pre requisite. Some of these resources contain bits of repeatitive information but combined they serve as a complete guide.

Positional Encodings are used to represent the index of the token in the sentences.

- <u>[Blog](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/)</u> introducing Positional Encoding in Transformers &rarr; *By Jason Brownlee*
- <u>[Discussion](https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model)</u> giving a short visual explanation about Positional Encodings &rarr; *Stack Overflow*
- <u>[Video](https://www.youtube.com/watch?v=dichIcUZfOw)</u> explaining Positional Encoding in detail &rarr; *By Batool Haider*
- <u>[Blog](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)</u> discussing the mathematics behind Positional Encoding. None of the resources on web go into such mathematical detail as much as this precious blog &rarr; *By Amirhossein Kazemnejad*


### Dependent Concepts

These ideas are necessary to implement any large Deep Learning model in general.

#### <u>Regularization</u>

Dropout is used to regularize the translation model in this architecture. It is easier to understand Dropout once the basic usage of regularization is clear.

- <u>[Video](https://www.youtube.com/watch?v=6g0t3Phly2M&t=1s)</u> explains what is regularization and L2 regularization in particular? &rarr; *By Andrew NG*
- <u>[Video](https://www.youtube.com/watch?v=NyG-7nRpsW8)</u> explains why regularization works intuitively &rarr; *By Andrew NG*


#### <u>Dropout</u>

Dropout is to prevent over-fitting the model on training data.

- <u>[Video](https://www.youtube.com/watch?v=D8PJAL-MZv8)</u> explains what Dropout is &rarr; *By Andrew NG*
- <u>[Video](https://www.youtube.com/watch?v=ARq74QuavAo)</u> explains why Dropout works intuitively &rarr; *By Andrew NG*


#### <u>Batch Normalization</u>

Batch Normalization is not used in this translation model architecture, however it provides a strong base to understand Layer Normalization better which is used in the model.

- <u>[Video1](https://www.youtube.com/watch?v=tNIpEZLv_eg)</u>, <u>[Video2](https://www.youtube.com/watch?v=em6dfRxYkYU)</u> explain what Batch Normalization is &rarr; *By Andrew NG*
- <u>[Video](https://www.youtube.com/watch?v=nUUqwaxLnWs)</u> explains why Batch Normalization works intuitively &rarr; *By Andrew NG*
- <u>[Blog](https://leimao.github.io/blog/Batch-Normalization/)</u> explains the math behind Batch Normalization &rarr; *By Lei Mao*
- <u>[Blog](https://gab41.lab41.org/batch-normalization-what-the-hey-d480039a9e3b)</u> takes a deeper look at Batch Normalization &rarr; *By Karl N*
- <u>[Video](https://youtu.be/OioFONrSETc?si=gcYbRgaP64D1rX8P)</u> explains the Batch Normalization paper &rarr; *By Yannic Kilcher*


#### <u>Layer Normalization</u>

Layer Normalization is used to speed up the learning process during training.

- <u>[Blog](https://www.kaggle.com/code/halflingwizard/how-does-layer-normalization-work)</u> explains what Layer Normalization is &rarr; *By Matt Namvarpour*
- <u>[Video](https://www.youtube.com/watch?v=2V3Uduw1zwQ&t=103s)</u> explains Layer Normalization by runnning it on an example &rarr; *By AssemblyAI*
- <u>[Blog](https://leimao.github.io/blog/Layer-Normalization/)</u> explains the math behing Layer Normalization &rarr; *By Lei Mao*


### Data Preparation

The (English - Telugu) translation pairs need to be tokenized and converted into the format needed by the translation model. The following resources help us achieve this.


#### <u>Translation Dataset</u>

This repository builds a model that translates English sentences to Telugu. We use the AI4Bharat Samantar dataset hosted on Hugging Face for our purposes.

- <u>[Blog](https://huggingface.co/datasets/ai4bharat/samanantar)</u> contains the translation dataset between English and 11 different Indian languages.


#### <u>Tokenization</u>

Tokenization is the process of dividing sentences into smaller entities called tokens.

- <u>[Blog](https://huggingface.co/docs/transformers/tokenizer_summary)</u> gives a short summary of tokenization and its necessity &rarr; *By Hugging Face*


#### <u>Byte Pair Encoding</u>

Byte Pair Encoding is used in this implementation to tokenize the input sentences.

- <u>[Blog](https://www.reedbeta.com/blog/programmers-intro-to-unicode/)</u> provides an in-depth introduction to Unicode encoding &rarr; *By Nathan Reed*
    * Unicode encoding is used in the Byte level Byte Pair encoding algorithm which we used to tokenize the input in this repository.
- <u>[Video](https://www.youtube.com/watch?v=zduSFxRajkE)</u> gives a walk through of the Byte level Byte Pair Encoding algorithm &rarr; *By Andrej Karpathy*


### Model Training

#### <u>KL Divergence</u>

KL Divergence is used to calculate the loss between predicted output and the expected output in the translation model.

- <u>[Blog](https://encord.com/blog/kl-divergence-in-machine-learning/)</u> explains how KL Divergence is used in Machine Learning &rarr; *By Encord* 
- <u>[Blog](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)</u> explains the idea of KL Divergence in the context of Probability &rarr; *By Will Kurt*
- <u>[Blog](https://dibyaghosh.com/blog/probability/kldivergence.html)</u> explains the math behind KL Divergence and why it works &rarr; *By Dibya Ghosh*

#### <u>Label Smoothing</u>

Label Smoothing is used to prevent the model from becoming over-confident in its predictions. It's a kind of regularization technique that alters the expected output to prevent over-fitting.

- <u>[Video](https://www.youtube.com/watch?v=wmUiOAra_-M)</u> explains the concept of Label Smotthing &rarr; *By Neil Rhodes*
- <u>[Blog](https://towardsdatascience.com/label-smoothing-make-your-model-less-over-confident-b12ea6f81a9a)</u> gives a short explanation on label smoothing is implemented &rarr; *By Parthvi Shah*


#### <u>Optimizers</u>

Optimizers are used to compute the loss, calculate the gradients during back propagation and update the parameters of the model.

- <u>[Video1](https://www.youtube.com/watch?v=lAq96T8FkTw&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=18)</u>, <u>[Video2](https://www.youtube.com/watch?v=NxTFlzBjS-4&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=19)</u>, <u>[Video3](https://www.youtube.com/watch?v=lWzo8CajF5s&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=19)</u> explain the concept of Exponentially weighted averages &rarr; *By Andrew NG*
- <u>[Video](https://www.youtube.com/watch?v=k8fTYJPd3_I&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=20)</u> explains gradient descent with momentum &rarr; *By Andrew NG*
- <u>[Video](https://www.youtube.com/watch?v=_e-LFe_igno&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=22)</u> explains the RMS prop algorithm &rarr; *By Andrew NG*
- <u>[Video](https://www.youtube.com/watch?v=JXQT_vxqwIs&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=22)</u> explains Adam Optimization which is used in this model &rarr; *By Andrew NG*


#### <u>Learning Rates</u>

Learning rate is used to control the size of updates to the weights during back propagation.

- <u>[Blog](https://www.jeremyjordan.me/nn-learning-rate/)</u> presents an excellent deep-dive on using learning rates &rarr; *By Jeremy Jordan*
- <u>[Video](https://www.youtube.com/watch?v=QzulmoOg2JE&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc&index=23)</u> explains the concept of learning rate decay &rarr; *By Andrew NG*
- <u>[Blog](https://cs231n.github.io/neural-networks-3/?ref=jeremyjordan.me#annealing-the-learning-rate)</u> presents a very detailed overview of training and learning rates for Neural Networks &rarr; *By Andrej Karpathy*


#### <u>Weight Initialization</u>

The trainable parameters of the model are initialized using Xavier Initialization to aid the process of finding an optimial minimum.

- <u>[Video](https://www.youtube.com/watch?v=8krd5qKVw-Q)</u> explains how weight initialization affects the neural network training and how Xavier initialization fixes these issues &rarr; *By deeplizard channel*
-<u>[Blog](https://www.deeplearning.ai/ai-notes/initialization/index.html)</u> presents a mathematical argument on why Xavier initialization works and tools to visualize the same &rarr; *By Deeplearning.ai*

### Model Inference

#### <u>Beam Search</u>

Greedy search selects the token with the maximum probability as the output of the model. Beam Search is an alternative to Greedy search which holds multiple tokens as potential output at each step and maximizes the probability at the end of the sequence. Beam search is only used during inference and not used during training.

- <u>[Video](https://www.youtube.com/watch?v=RLWuzLLSIgw)</u> gives an intuitive explanation of Beam Search &rarr; *By Andrew NG*
- <u>[Video](https://www.youtube.com/watch?v=gb__z7LlN_4)</u> explains possible improvements to Beam Search &rarr; *By Andrew NG*
- <u>[Video](https://www.youtube.com/watch?v=ZGUZwk7xIwk)</u> explains how error analysis is performed on Beam Search &rarr; *By Andrew NG*

### Implementation

The above resources help us understand the model architecture and necessary dependencies in detail. However, there are still several gaps wrt the implementation details. This section presents the resources that aid in the implementation of the translation model.

#### <u>Existing Implementations for Reference</u>

There are several great existing implementations of the *Attention Is All You Need* paper. I used some of these resources to understand the implementation and make my own version.

- <u>[Blog](https://nlp.seas.harvard.edu/annotated-transformer/)</u> gives a very detailed explanation of the model implementation &rarr; *By Harvard team*
    * I picked the core model implementation from this blog, ran pieces of it in my Jupiter notebooks (uploaded to this repo), explained every single line of code and used it in my final implementation. Made changes whenever necessary to suit my repository.
    * This is an excellent resource but not very beginner friendly and takes considerable effort to understand the code. 
- <u>[Github Repo](https://github.com/gordicaleksa/pytorch-original-transformer)</u> is useful to understand some of the implementation aspects related to the training and inference &rarr; *By Aleksa Gordic* 


#### <u>Understanding Implementation Details</u>

- <u>[Blog](https://peterbloem.nl/blog/transformers)</u> is very useful in understanding the transformations that the input undergoes during the Multi Headed Attention layer &rarr; *By Peter Bloem*
- <u>[Video](https://www.youtube.com/watch?v=cbYxHkgkSVs)</u> gives a walk through of the input transformation on an example &rarr; *By Aleksa Gordic*
- <u>[Video](https://www.youtube.com/watch?v=IGu7ivuy1Ag)</u> gives a walk through on how the model behavior varies during training and inference &rarr; *By Niels Rogge*
    * This is an extremely useful resource. Most of the explanations on web overlook this part which is required to be understood to implement the model.
- <u>[Video](https://www.youtube.com/watch?v=dZzVA6VbAR8)</u> explains the data preparation process (first 30 minutes in the video) for the transformers &rarr; *By Ana Marasovic*
    * This is one more such extremely useful resource which is over looked in most of the other explanations of transformer model. Data Preparation turned out to be one of the most challenging aspects in building this repository.


#### <u>Pytorch</u>

- <u>[Github repo]()</u> explains in detail with examples how the common functions in pytorch work &rarr; *By Maneesh Babu Adhikari*

#### <u>Using GPU</u>

- <u>[Video](https://youtu.be/6stDhEA0wFQ?si=rkc0iKKRxWnaYbYo)</u> explains why GPUs are used for training and inference in Deep Learning &rarr; *By deeplizard channel*
- <u>[Video](https://youtu.be/Bs1mdHZiAS8?si=0SpkfO3POIuffsv3)</u> explains how to create tensors and modules on GPU using pytorch &rarr; *By deeplizard channel*


## Repository Structure

- `building_transformers_step_by_step/` &rarr; Contains jupyter notebooks with detailed explanations of each step in the implementation.
- `Data/` &rarr; Contains the datasets used for training, generated artifacts by the model, 
- `model_implementation/` &rarr; Holds the modularized python scripts for training and inference. 


### Deep Dive into `building_transformers_step_by_step/`

This directory houses a collection of Jupyter notebooks that break down the translation model implementation into manageable chunks. Each notebook focuses on a specific aspect, providing a step-by-step walkthrough of the code along with in-depth explanations.

Here's a short overview of what you'll find inside:

#### `data_preparation/`

- `step_1_data_exploration.ipynb`
    * Loads the AI4Bharat Samanantar English-Telugu translation dataset.
    * Explores the structure of the loaded data.
    * Creates repo specific datasets and saves to `Data/AI4Bharat`.
- `step_2_training_bpe_tokenizer.ipynb`
    * Uses the HuggingFace libraries to train a tokenizer using the Byte level Byte Pair Encoding (BPE) algorithm on the training data and explores the results.
    * Saves the trained tokenizers to `Data/trained_models/tokenizers`.
- `step_2_alternate_tokenization_with_spacy.ipynb`
    * Explores an alternate tokenization mechanism with pretrained spacy tokenizers.
- `step_3_datasets_and_dataloaders_pytorch.ipynb`
    * Explores the creation of general pytorch DataLoaders to load the data efficiently.
- `step_4_dataloader_with_transformers.ipynb`
    * Explains how different datasets can be integrated with Pytorch DataLoaders.
    * Creates pytorch Dataloaders to load the English-Telugu translation dataset.
    * Creates length aware pytorch Dataloaders that batch the data based on the sentence lengths.
- `step_5_data_batching_and_masking.ipynb`
    * Shows how to create masks for the source and target sequences.
    * Shows how the batches from pytorch Dataloaders are futher processed to convert the data into the format required by the transformer model.


#### `model_building/`

- `step_6_word_embeddings.ipynb`
    * Explores how Embedding layers are created and used in pytorch for tokens in general.
    * Shows how the Embedding layer is used in the translation model.
- `step_7_drop_out.ipynb`
    * Explains what dropout is, how it is used and the impact of Dropout the output.
- `step_8_positional_encoding.ipynb`
    * Explains the creation of positional encoding vectors and how they are used in the translation model.
- `step_9_multi_headed_attention.ipynb`
    * Explains how to implement Attention module in the transformer architecture.
    * Explains how to implement multi headed attention efficiently in one shot as if handling a single head instead of handling different heads separately.
- `step_10_feed_forward_neural_network.ipynb`
    * Explains how FeedForward neural network is used in translation model.
- `step_11_layer_normalization.ipynb`
    * Explains how layer normalization works in general and how it is used in the translation model in particular.
- `step_12_encoder.ipynb`
    * Explains how Encoder works and its implementation in the translation model.
- `step_13_decoder.ipynb`
    * Explains how Decoder works and its implementation in the translation model.
- `step_14_token_predictor.ipynb`
    * Explains the last layer in the translation model that converts the output of the Decoder into probability distributions over the target vocabulary space.


#### `model_training_and_inference/`

- `step_16_label_smoothing.ipynb`
    * Explains what label smoothing is, how it is used in general and how it is used in the translation model.
- `step_17_loss_computation.ipynb`
    * Explains what KL Divergence is, how it is used to compute the loss in general and how it is used in the translation model.
- `step_18_learning_rates.ipynb`
    * Explains how learning rate is used to control the training speed in the translation model.
- `step_19_model_training.ipynb`
    * Combines all the moving parts to create a module to train the translation model.
- `step_20_beam_search.ipynb`
    * Explains how Beam Search is used to predict the model output during inference.


### Deep Dive into `Data/`

This directory houses all the resources related to data used by the model and additional information to understand the model arrchitecture. This includes the raw datasets, artifacts generated during data processing, the trained models and resources to aid the understanding of the model architecture.

#### `AI4Bharat/`

All the datasets in this directory get generated once the `building_transformers_step_by_step/data_preparation/step_1_data_exploration.ipynb` notebook is fully run.

- `debug_dataset`
    * A very short dataset used for sample training and identifying issues / bugs in the model implementation.
- `train_dataset`
    * Dataset used to train the translation model.
- `full_en_te_dataset`
    * Dataset used to train the tokenizers using BPE.
- `validation_dataset`
    * Dataset used to validate the model periodically during the training.


#### `Images/`

Contains the images used in various notebooks to explain the model architecture visually.


#### `Resources/`

Contains additional resources that help in understanding the model architecture.

- `Input_Transformation_In_Multi_Headed_Attention.pdf`
    * Shows visually the input transformation during the multi headed attention operation.

#### `trained_models/`

- `tokenizers`
    * Holds the trained BPE English and Telugu language tokenizers.
- `translation_models`
    * Holds the model checkpoint that are periodically created in between the training.


### Deep Dive into `model_implementation/`

Model Implementation houses the modularized version of the same translation model that is implemented in the Jupiter notebooks. It just has better structure and easy to navigate. However, `model_implementation/` contains additional scripts for training and inference which are not part of the notebooks.  

- `model_debug_support.ipynb`
    * A support notebook to run parts of the code and figure out issues / bugs in the model implementation.


## Usage

### Set Up

Create a Virtual Environment for this project that will contain all the dependencies.

```python -m venv .attention_venv```

Run the following command to install the necessary packages in the virtual environment.

```pip install -r requirements.txt```

### Training

The entry point to train the model is `model_implementation/model_training/training_script_main.py`

The training script accepts the following command line arguments:

```
- model_checkpoint_prefix (Optional): Prefix to be appended to model names while saving to disk. Defaults to empty string ("").
- model_name (Optional): Name to use to save the final model on disk. Defaults to a randomly generated string.
- device (Optional): Device to be used for training the model. Can be 'cpu' or 'cuda'. Defaults to 'cpu'.
- retrain_tokenizers (Optional): Flag to indicate if the tokenizers should be retrained. Defaults to False
- max_english_vocab_size (Optional): Maximum size of the English vocabulary. Only used if retrain_tokenizers is set to True. Defaults to 30000.
- max_telugu_vocab_size (Optional): Maximum size of the Telugu vocabulary. Only used if retrain_tokenizers is set to True. Defaults to 30000.
```

Run the following command to train the model:

```
python model_implementation/model_training/training_script_main.py --model_name "en-te-model" --model_checkpoint_prefix "first_run" --device "cuda" --retrain_tokenizers False
```

### Inference

Inference script will load the trained model into memory and translate any input English sentences into Telugu. Unfortunately, the script does not have any guard rails. It will try to translate the sentences of any language and output garbage if the input is not in English.

The entry point to model inference is `model_implementation/model_inference/inference_script_main.py`

The inference script accepts the following command line arguments:

```
- model_name (Required): Name of the model to load from disk.
- model_checkpoint_prefix (Optional): Prefix to be appended to model names while loading from the disk. Defaults to empty string ("").
- beam_width (Optional): Width of the beam to be used in the beam search algorithm. Defaults to 3.
- device (Optional): Device to be used during model inference. Can be 'cpu' or 'cuda'. Defaults to 'cpu'.
```

Run the following command to use the trained model for inference:

```
python model_implementation/model_inference/inference_script_main.py --model_name "en-te-model" --model_checkpoint_prefix "first_run" --device "cuda" --beam_width 3
```

## Hardware

I trained the model on `RTX 4090, 16GB GPU` machines. These are the stats for the training:
