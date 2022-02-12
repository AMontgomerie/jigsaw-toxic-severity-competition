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
