import argparse
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List

from dataset import ToxicDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--save_path", type=str, default="./submission.csv")
    parser.add_argument("--base_model", type=str, default="roberta-base")
    parser.add_argument("--base_model_name", type=str, default="roberta-base")
    parser.add_argument("--weights_dir", type=str, default=".")
    parser.add_argument("--data_path", type=str, default="data/valid.csv")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dataloader_workers", type=int, default=2)
    return parser.parse_args()


@torch.no_grad()
def predict(
    model: AutoModelForSequenceClassification,
    dataset: Dataset,
    name: str,
    batch_size: int,
    dataloader_workers: int,
) -> List[float]:
    model.eval()
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataloader_workers,
        pin_memory=True,
    )
    predictions = []
    with tqdm(total=len(dataloader), unit="batches") as tepoch:
        tepoch.set_description(name)
        for data in dataloader:
            data = {k: v.to("cuda") for k, v in data.items()}
            output = model(**data)
            predictions += list(output.logits.squeeze().cpu().numpy())
            tepoch.update(1)
    return predictions


if __name__ == "__main__":
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1
    )
    model = model.to("cuda")
    data = pd.read_csv(args.data_path)
    dataset = ToxicDataset(data.text, tokenizer, args.max_length)
    scores = []
    for fold in range(args.num_folds):
        weights_path = os.path.join(
            args.weights_dir, f"{args.base_model_name.replace('/', '_')}_{fold}.bin"
        )
        print(f"Loading {weights_path}.")
        state_dict = torch.load(weights_path, map_location=torch.device("cuda"))
        model.load_state_dict(state_dict)
        ranking_score = predict(
            model,
            dataset,
            f"{args.base_model_name} fold {fold}",
            args.batch_size,
            args.dataloader_workers,
        )
        scores.append(ranking_score)
    mean_scores = np.mean(scores, axis=0)
    submission = pd.DataFrame({"comment_id": data.comment_id, "score": mean_scores})
    submission.to_csv(args.save_path, index=False)
