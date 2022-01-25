import pandas as pd
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dataset import ToxicDataset
from inference import predict, parse_args


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1
    )
    model = model.to("cuda")
    data = pd.read_csv(args.data_path)
    data["less_toxic_score"] = 0
    data["more_toxic_score"] = 0
    for fold in range(args.num_folds):
        weights_path = os.path.join(
            args.weights_dir, f"{args.base_model_name.replace('/', '_')}_{fold}.bin"
        )
        print(f"Loading {weights_path}.")
        state_dict = torch.load(weights_path, map_location=torch.device("cuda"))
        model.load_state_dict(state_dict)
        if args.num_folds > 1:
            fold_data = data.loc[data.fold == fold].reset_index(drop=True)
        else:
            fold_data = data
        less_toxic_dataset = ToxicDataset(
            fold_data.less_toxic, tokenizer, args.max_length
        )
        less_toxic_preds = predict(
            model,
            less_toxic_dataset,
            f"{args.base_model_name} fold {fold} less toxic",
            args.batch_size,
            args.dataloader_workers,
        )
        more_toxic_dataset = ToxicDataset(
            fold_data.more_toxic, tokenizer, args.max_length
        )
        more_toxic_preds = predict(
            model,
            more_toxic_dataset,
            f"{args.base_model_name} fold {fold} more toxic",
            args.batch_size,
            args.dataloader_workers,
        )
        if args.num_folds > 1:
            data.loc[data.fold == fold, "less_toxic_score"] = less_toxic_preds
            data.loc[data.fold == fold, "more_toxic_score"] = more_toxic_preds
        else:
            data.loc[:, "less_toxic_score"] = less_toxic_preds
            data.loc[:, "more_toxic_score"] = more_toxic_preds
    data.to_csv(args.save_path, index=False)
