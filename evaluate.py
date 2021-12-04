import argparse
import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import gc

from dataset import ToxicDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="roberta-base")
    parser.add_argument("--weights_dir", type=str, default=".")
    parser.add_argument("--data_path", type=str, default="data/valid.csv")
    parser.add_argument("--max_length", type=int, default=512)
    return parser.parse_args()


def evaluate(
    base_model: str, data: pd.DataFrame, weights_dir: str, max_length: int
) -> float:
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    less_toxic_dataset = ToxicDataset(data.less_toxic, tokenizer, max_length)
    more_toxic_dataset = ToxicDataset(data.more_toxic, tokenizer, max_length)
    less_toxic_dataloader = get_dataloader(less_toxic_dataset)
    more_toxic_dataloader = get_dataloader(more_toxic_dataset)
    less_toxic_predictions = []
    more_toxic_predictions = []
    for fold in range(5):
        model = load_model(base_model, weights_dir, fold)
        less_toxic = predict(model, less_toxic_dataloader)
        more_toxic = predict(model, more_toxic_dataloader)
        less_toxic_predictions.append(less_toxic)
        more_toxic_predictions.append(more_toxic)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    mean_less_toxic = np.mean(less_toxic_predictions, axis=0)
    mean_more_toxic = np.mean(more_toxic_predictions, axis=0)
    return sum(mean_less_toxic < mean_more_toxic) / len(data)


@torch.no_grad()
def predict(
    model: AutoModelForSequenceClassification, dataloader: DataLoader
) -> np.ndarray:
    model.eval()
    predictions = np.array([])
    with tqdm(total=len(dataloader), unit="batches") as tepoch:
        tepoch.set_description("evaluation")
        for data in dataloader:
            data = {k: v.to("cuda") for k, v in data.items()}
            output = model(**data)
            np.append(predictions, output.logits.cpu().numpy())
            tepoch.update(1)
    return predictions


def load_model(
    base_model: str, save_dir: str, fold: int
) -> AutoModelForSequenceClassification:
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)
    weights_path = os.path.join(save_dir, f"{base_model.replace('/', '_')}_{fold}.bin")
    model.load_state_dict(torch.load(weights_path))
    model = model.to("cuda")
    return model


def get_dataloader(dataset: Dataset) -> DataLoader:
    return DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True
    )


if __name__ == "__main__":
    args = parse_args()
    data = pd.read_csv(args.data_path)
    ranking_score = evaluate(args.base_model, data, args.weights_dir, args.max_length)
    print(ranking_score)