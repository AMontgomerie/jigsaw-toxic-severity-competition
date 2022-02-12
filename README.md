# jigsaw-toxic-severity-competition

This repo contains the source code for the 14th place solution in the [Jigsaw Rate Severity of Toxic Comments competition](https://www.kaggle.com/c/jigsaw-toxic-severity-rating) on Kaggle. A write-up of the solution can be found on my blog [here](https://amontgomerie.github.io/2022/02/08/jigsaw-toxic-severity-competition.html).

## Fine-Tuning

Here's an example of fine-tuning roberta-base on ruddit:

```
python train.py \
    --group_id 1 \
    --checkpoint roberta-base \
    --train_path ruddit.csv \
    --valid_path validation_folds.csv \
    --save_dir ./ \
    --accumulation_steps 1 \
    --train_batch_size 64 \
    --valid_batch_size 256 \
    --max_length 128 \
    --epochs 20 \
    --learning_rate 1e-6 \
    --scheduler linear \
    --warmup 0.05 \
    --early_stopping_patience 5
```

`group_id` is only used to associate this run with a group in weights&biases. This will use the whole of `ruddit.csv` as training data, and the whole of `validation_folds.csv` as validation data (ignoring the folds).

`--early_stopping_patience` is set here so that the training can terminate early if necessary. Only the epoch which gets the best score on the validation set will be saved.

Now here's an example of a second stage of fine-tuning using the model we just trained on ruddit. This time we're using the paired validation data and training with MarginRankingLoss instead of MSELoss:

```
python paired_train.py \
    --group_id 2 \
    --checkpoint roberta-base \
    --fold 0 \
    --train_path validation_folds.csv \
    --save_dir ./ \
    --accumulation_steps 1 \
    --train_batch_size 64 \
    --valid_batch_size 256 \
    --max_length 128 \
    --epochs 5 \
    --learning_rate 1e-6 \
    --scheduler linear \
    --warmup 0.05 \
    --save_all \
    --weights_path ./roberta-base_0.bin
```

Note that in this case `validation_folds.csv` needs to have an extra column called `fold` which will be use in combination with the `--fold` argument to split the data for training and validation. This script will need to be run k times with each value of `--fold` to produce the full cross-validated set of models.

`--weights_path` is optional. If provided then the given weights file we be used to initialise the model, otherwise the training will start from the pretrained checkpoint defined by `--checkpoint`.

Instead of using `--early_stopping_patience`, we use `--save_all` here. This saves a checkpoint for each epoch (without overwriting the previous one). This allows us to calculate a CV score for each epoch and take the checkpoints from the epoch which did the best (we can then discard the weights from all other epochs).

See `utils.parse_training_args` for the full list of command line options.

## Ensembling

A notebook containing the code for tuning ensemble weights and also the final inference notebook have also been included. These notebooks have simply been copied from Kaggle so paths to data files and model weights will of course need to be modified to make them usable.

### tune-ensemble-weights

In this notebook, each (k-fold) model is represented by an OOF file containing its predictions on the validation data. The predictions are blended by calculating a weighted mean. The weights for each model are determined by Optuna. The model selection process takes place over a series of rounds. In each round each available model is temporarily added to the ensemble and all the weights are retuned. Whichever additional model produced the highest increase in OOF score will be added to the ensemble. This process is repeated until the score stops improving.

### ensemble-inference

In this notebook, the inference script is run for each model in turn. The inference sscript outputs the predictions to a CSV file. The CSV files are then all read back into the notebook, and the model weights produced in `tune-ensemble-weights` are used to calculate a weighted mean for each prediction.
