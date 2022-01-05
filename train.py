import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import wandb

from utils import set_seed, parse_training_args
from dataset import ToxicDataset
from trainer import Trainer
from model import convert_regressor_to_binary

if __name__ == "__main__":
    args = parse_training_args()
    config = vars(args)
    if config["use_extra_data"]:
        extra_files = [
            os.path.join(config["extra_data_dir"], f)
            for f in os.listdir(config["extra_data_dir"])
        ]
        config["extra_files"] = extra_files
    wandb.login()
    if config["num_labels"] is None or config["num_labels"] == 1:
        project = "jigsaw-train"
    else:
        project = "jigsaw-binary-train"
    with wandb.init(
        project=project,
        group=str(args.group_id),
        name=f"{args.group_id}-{args.checkpoint}",
        config=config,
    ):
        config = wandb.config
        set_seed(config.seed)
        train_data = pd.read_csv(config.train_path)
        if config.use_extra_data:
            extra_data = [pd.read_csv(f) for f in extra_files]
            train_data = pd.concat([train_data] + extra_data)
        valid_data = pd.read_csv(config.valid_path)
        tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.checkpoint, num_labels=config.num_labels
        )
        if config.weights_path is not None:
            state_dict = torch.load(
                config.weights_path, map_location=torch.device("cuda")
            )
            if config.num_labels == 2:
                state_dict = convert_regressor_to_binary(state_dict)
            model.load_state_dict(state_dict)
        train_set = ToxicDataset(
            train_data.text, tokenizer, config.max_length, train_data.target
        )
        less_toxic_valid_set = ToxicDataset(
            valid_data.less_toxic, tokenizer, config.max_length
        )
        more_toxic_valid_set = ToxicDataset(
            valid_data.more_toxic, tokenizer, config.max_length
        )
        trainer = Trainer(
            accumulation_steps=config.accumulation_steps,
            dataloader_workers=config.dataloader_workers,
            early_stopping_patience=config.early_stopping_patience,
            epochs=config.epochs,
            fold=config.fold if config.fold else 0,
            learning_rate=config.learning_rate,
            less_toxic_valid_set=less_toxic_valid_set,
            log_interval=config.log_interval,
            loss_type=config.loss_type,
            model=model,
            model_name=config.checkpoint,
            more_toxic_valid_set=more_toxic_valid_set,
            num_labels=config.num_labels,
            save_dir=config.save_dir,
            scheduler=config.scheduler,
            train_batch_size=config.train_batch_size,
            train_set=train_set,
            valid_batch_size=config.valid_batch_size,
            validation_steps=config.validation_steps,
            warmup=config.warmup,
            weight_decay=config.weight_decay,
        )
        trainer.train()