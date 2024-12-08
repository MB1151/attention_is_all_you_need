# Training Statistics

This file contains the following runtime statistics of the model during training. You can use this to get an idea on how the learning rates, losses, ... etc vary over time during your training loops.

The following details are logged after epoch during training:

- Epoch Number
- Training time for that specific epoch
- Training loss at the end of the epoch
- Number of tokens processed during the epoch
    - This might vary in every epoch because of the way the batches are created.
- Learning rate at the start of the epoch
- Learning rate at the end of the epoch
- Model checkpoint path
    - The path to the saved model at the end of the epoch.
    - There will be 1 model instance per epoch.
- Number of batches skipped
    - Batches are skipped when the number of tokens in the longest sentence for that particular batch exceeds 150.


## Model Parameters

The full list of model parameters used to train this translation model can be found in [`constants.py`](model_implementation/utils/constants.py) file.

However, the most important parameters are:

```
Batch Size = 64
Dataset Size = 500000 English - Telugu translation pairs
Tokenizer = Bype Pair Encoding or bpe
English Vocabulary Size = 30000
Telugu Vocabulary Size = 30000
```

## Runtime Statistics

```
[2024-12-07 21:11:49,483 -- __main__ -- INFO -- Model training completed in 527.4490317424138 minutes.]

[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Device used for training: cuda]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Epoch: 0]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 27.109612290064494]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 1.746928107421711e-07]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.0005289397437729705]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 2.5292908727308383]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 16889921]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_0_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 832]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 1.3851797957474814]
---------------------------------------------------------------------------

---------------------------------------------------------------------------
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Epoch: 1]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 27.10041936635971]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 0.0005289397437729705]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.0003736824761467883]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 1.2983434772150908]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 16959483]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_1_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 807]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 1.1253165906611402]
---------------------------------------------------------------------------

---------------------------------------------------------------------------
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Epoch: 2]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 25.55793274641037]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 0.0003736824761467883]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.00030402918197545525]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 1.1227914585902943]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 17185279]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_2_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,484 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 670]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 1.0506169486259083]
---------------------------------------------------------------------------

---------------------------------------------------------------------------
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Epoch: 3]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 25.31948359409968]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 0.00030402918197545525]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.00026274887799742285]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 1.034159839440593]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 17166273]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_3_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 652]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 0.9816841645535159]
---------------------------------------------------------------------------

---------------------------------------------------------------------------
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Epoch: 4]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 24.63714582125346]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 0.00026274887799742285]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.00023449557638210213]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 0.9772498029929141]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 17276909]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_4_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 585]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 0.974618331162907]
---------------------------------------------------------------------------

---------------------------------------------------------------------------
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Epoch: 5]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 25.23422723611196]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 0.00023449557638210213]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.00021394576693215602]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 0.9373735969038851]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 17143788]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_5_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 662]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 0.9420654765753859]
---------------------------------------------------------------------------

---------------------------------------------------------------------------
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Epoch: 6]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 25.024186352888744]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 0.00021394576693215602]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.00019787995234813236]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 0.9043254315773664]
[2024-12-07 21:11:49,485 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 17263723]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_6_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 603]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 0.9337747716685599]
---------------------------------------------------------------------------

---------------------------------------------------------------------------
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Epoch: 7]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 24.76220065355301]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 0.00019787995234813236]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.00018496791839503303]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 0.8770807410666143]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 17233626]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_7_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 606]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 0.9186003275378982]
---------------------------------------------------------------------------

---------------------------------------------------------------------------
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Epoch: 8]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 27.368804546197257]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 0.00018496791839503303]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.0001746791644654841]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 0.859651370810228]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 16764191]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_8_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 890]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 0.9063842863274355]
---------------------------------------------------------------------------

---------------------------------------------------------------------------
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Epoch: 9]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 25.972533174355824]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 0.0001746791644654841]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.00016572362039730433]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 0.8365401450092991]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 17115208]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_9_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 708]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 0.8869720375672059]
---------------------------------------------------------------------------

---------------------------------------------------------------------------
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Epoch: 10]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 26.820706872145333]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 0.00016572362039730433]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.00015813108860689487]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 0.8203578119928925]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 16889520]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_10_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 820]
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 0.8786564146501775]
---------------------------------------------------------------------------

---------------------------------------------------------------------------
[2024-12-07 21:11:49,486 -- model_implementation.utils.state_holders -- INFO -- Epoch: 11]
[2024-12-07 21:11:49,487 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 26.0411194284757]
[2024-12-07 21:11:49,487 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 0.00015813108860689487]
[2024-12-07 21:11:49,487 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.00015141829702841603]
[2024-12-07 21:11:49,487 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 0.8032269145931444]
[2024-12-07 21:11:49,487 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 17033101]
[2024-12-07 21:11:49,487 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_11_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,487 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 734]
[2024-12-07 21:11:49,487 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 0.8878250299568463]
---------------------------------------------------------------------------

---------------------------------------------------------------------------
[2024-12-07 21:11:49,487 -- model_implementation.utils.state_holders -- INFO -- Epoch: 12]
[2024-12-07 21:11:49,487 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 25.75794781843821]
[2024-12-07 21:11:49,487 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 0.00015141829702841603]
[2024-12-07 21:11:49,487 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.00014549054114101122]
[2024-12-07 21:11:49,487 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 0.7888535774450467]
[2024-12-07 21:11:49,487 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 17039038]
[2024-12-07 21:11:49,487 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_12_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,487 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 730]
[2024-12-07 21:11:49,498 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 0.8805355780906867]
---------------------------------------------------------------------------

---------------------------------------------------------------------------
[2024-12-07 21:11:49,498 -- model_implementation.utils.state_holders -- INFO -- Epoch: 13]
[2024-12-07 21:11:49,498 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 27.718947839736938]
[2024-12-07 21:11:49,498 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 0.00014549054114101122]
[2024-12-07 21:11:49,498 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.00014032725202876614]
[2024-12-07 21:11:49,498 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 0.7795435474194353]
[2024-12-07 21:11:49,498 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 16777013]
[2024-12-07 21:11:49,498 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_13_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,498 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 898]
[2024-12-07 21:11:49,498 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 0.8703714229739113]
---------------------------------------------------------------------------

---------------------------------------------------------------------------
[2024-12-07 21:11:49,498 -- model_implementation.utils.state_holders -- INFO -- Epoch: 14]
[2024-12-07 21:11:49,498 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 26.56455702781677]
[2024-12-07 21:11:49,498 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 0.00014032725202876614]
[2024-12-07 21:11:49,498 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.00013561536442313464]
[2024-12-07 21:11:49,498 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 0.7651021524782885]
[2024-12-07 21:11:49,498 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 16922593]
[2024-12-07 21:11:49,498 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_14_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,498 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 801]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 0.8687984162224982]
---------------------------------------------------------------------------

---------------------------------------------------------------------------
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Epoch: 15]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 24.209542644023895]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 0.00013561536442313464]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.00013121221107629882]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 0.7497387390106833]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 17298227]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_15_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 566]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 0.8675131615975157]
---------------------------------------------------------------------------

---------------------------------------------------------------------------
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Epoch: 16]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 25.599616066614786]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 0.00013121221107629882]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.00012727514135793934]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 0.7439651315678193]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 17135930]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_16_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 686]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 0.8556179970807332]
---------------------------------------------------------------------------

---------------------------------------------------------------------------
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Epoch: 17]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 24.239453097184498]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 0.00012727514135793934]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.00012361390090994298]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 0.7309681836205462]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 17300057]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_17_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 565]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 0.8695457193470684]
---------------------------------------------------------------------------

---------------------------------------------------------------------------
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Epoch: 18]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 26.075454795360564]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 0.00012361390090994298]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.00012032810944838252]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 0.7276885953597754]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 17036708]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_18_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 737]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 0.8673696699090111]
---------------------------------------------------------------------------

---------------------------------------------------------------------------
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Epoch: 19]
[2024-12-07 21:11:49,499 -- model_implementation.utils.state_holders -- INFO -- Training Time (in minutes): 26.549733662605284]
[2024-12-07 21:11:49,500 -- model_implementation.utils.state_holders -- INFO -- Epoch start learning Rate: 0.00012032810944838252]
[2024-12-07 21:11:49,500 -- model_implementation.utils.state_holders -- INFO -- Epoch end learning Rate: 0.00011732089085893435]
[2024-12-07 21:11:49,500 -- model_implementation.utils.state_holders -- INFO -- Training Loss: 0.7193146627471656]
[2024-12-07 21:11:49,500 -- model_implementation.utils.state_holders -- INFO -- Num Tokens Processed: 16892155]
[2024-12-07 21:11:49,500 -- model_implementation.utils.state_holders -- INFO -- Model Checkpoint Path: /content/drive/My Drive/Learning AI/Artificial Intelligence/Projects/Github/attention_is_all_you_need/Data/trained_models/translation_models/colab_run_3_epoch_19_bpe_large_train_dataset.pt]
[2024-12-07 21:11:49,500 -- model_implementation.utils.state_holders -- INFO -- Number of batches skipped: 809]
[2024-12-07 21:11:49,500 -- model_implementation.utils.state_holders -- INFO -- Validation Loss: 0.8681228350089374]
```