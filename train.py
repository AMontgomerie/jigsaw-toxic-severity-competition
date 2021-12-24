import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import wandb

from utils import set_seed, parse_training_args
from dataset import ToxicDataset
from trainer import Trainer

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
    with wandb.init(
        project="jigsaw-train",
        group=str(args.group_id),
        name=f"{args.group_id}-{args.checkpoint}",
        config=config,
    ):
        config = wandb.config
        set_seed(config.seed)
        train_data = pd.read_csv(config.train_path)
        valid_data = pd.read_csv(config.valid_path)
        tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(
            config.checkpoint, num_labels=config.num_labels
        )
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
            model=model,
            model_name=config.checkpoint,
            more_toxic_valid_set=more_toxic_valid_set,
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