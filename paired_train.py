import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

from utils import set_seed, parse_training_args
from dataset import ToxicDataset, PairedToxicDataset
from trainer import PairedTrainer
import wandb

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
    fold = args.fold if args.fold is not None else 0
    with wandb.init(
        project="jigsaw-paired-train",
        group=str(args.group_id),
        name=f"{args.group_id}-{args.checkpoint}-fold-{fold}",
        config=config,
    ):
        config = wandb.config
        set_seed(config.seed)
        data = pd.read_csv(config.train_path)
        if args.fold is not None:
            train_data = data.loc[data.fold != fold].reset_index(drop=True)
            valid_data = data.loc[data.fold == fold].reset_index(drop=True)
        else:
            train_data = pd.read_csv(config.train_path)
            valid_data = pd.read_csv(config.valid_path)
        if config.use_extra_data:
            for file in extra_files:
                print(file)
            extra_data = pd.concat([pd.read_csv(f) for f in extra_files])
            train_data = pd.concat([train_data, extra_data]).reset_index(drop=True)
        tokenizer = AutoTokenizer.from_pretrained(config.checkpoint)
        train_set = PairedToxicDataset(
            train_data.less_toxic, train_data.more_toxic, tokenizer, config.max_length
        )
        less_toxic_valid_set = ToxicDataset(
            valid_data.less_toxic, tokenizer, config.max_length
        )
        more_toxic_valid_set = ToxicDataset(
            valid_data.more_toxic, tokenizer, config.max_length
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            config.checkpoint, num_labels=1
        )
        if config.weights_path is not None:
            state_dict = torch.load(
                config.weights_path, map_location=torch.device("cuda")
            )
            print(f"Loading weights from {config.weights_path}")
            model.load_state_dict(state_dict)
        trainer = PairedTrainer(
            accumulation_steps=config.accumulation_steps,
            dataloader_workers=config.dataloader_workers,
            early_stopping_patience=config.early_stopping_patience,
            epochs=config.epochs,
            fold=fold,
            learning_rate=config.learning_rate,
            less_toxic_valid_set=less_toxic_valid_set,
            log_interval=config.log_interval,
            loss_margin=config.loss_margin,
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
        best_valid_score = trainer.train()
        print(f"Best valid score: {best_valid_score:.3f}")