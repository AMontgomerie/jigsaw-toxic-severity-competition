import argparse
import pandas as pd
import numpy as np
import spacy
import joblib
import os
from sklearn.base import BaseEstimator
from xgboost import XGBRegressor
from typing import List


class SpacyVectorizer:
    def __init__(self) -> None:
        self.tokenizer = spacy.load("en_core_web_lg")

    def transform(self, texts: pd.Series) -> List[np.ndarray]:
        print(f"Encoding {len(texts)} texts...")
        tokenized_texts = list(self.tokenizer.pipe(texts))
        text_vectors = []
        for text in tokenized_texts:
            vectors = [token.vector for token in text]
            mean_vector = np.mean(vectors, axis=0)
            text_vectors.append(mean_vector)
        return text_vectors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--valid_path", type=str, default="data/valid.csv")
    parser.add_argument("--test_path", type=str, default="data/comments_to_score.csv")
    parser.add_argument("--model_save_dir", type=str, default=".")
    parser.add_argument("--pred_save_path", type=str, default="./submission.csv")
    return parser.parse_args()


def train_all_folds(data: pd.DataFrame, save_dir: str) -> List[XGBRegressor]:
    models = []
    for fold in range(5):
        train = data.loc[data.fold != fold]
        model = XGBRegressor()
        model.fit(train.vector.tolist(), train.target)
        models.append(model)
        save_model(model, fold, save_dir)
    return models


def validate(
    models: List[XGBRegressor], less_toxic: pd.Series, more_toxic: pd.Series
) -> float:
    fold_less_toxic = []
    fold_more_toxic = []
    for model in models:
        less_toxic_preds = model.predict(less_toxic)
        more_toxic_preds = model.predict(more_toxic)
        fold_less_toxic.append(less_toxic_preds)
        fold_more_toxic.append(more_toxic_preds)
    mean_less_toxic = np.mean(fold_less_toxic, axis=0)
    mean_more_toxic = np.mean(fold_more_toxic, axis=0)
    return sum(mean_less_toxic < mean_more_toxic) / len(less_toxic)


def predict(models: List[XGBRegressor], test_data: pd.Series) -> np.ndarray:
    fold_predictions = []
    for model in models:
        predictions = model.predict(test_data)
        fold_predictions.append(predictions)
    return np.mean(fold_predictions, axis=0)


def save_model(model: BaseEstimator, fold: int, save_dir: str) -> None:
    save_path = os.path.join(save_dir, f"tfidf_ridge_fold_{fold}.pkl")
    joblib.dump(model, save_path)


if __name__ == "__main__":
    args = parse_args()
    data = pd.read_csv(args.train_path)
    valid_data = pd.read_csv(args.valid_path)
    test_data = pd.read_csv(args.test_path)
    encoder = SpacyVectorizer()
    data["vector"] = encoder.transform(data.text)
    encoded_less_toxic = encoder.transform(valid_data.less_toxic)
    encoded_more_toxic = encoder.transform(valid_data.more_toxic)
    encoded_test_data = encoder.transform(test_data.text)
    models = train_all_folds(data, args.model_save_dir)
    valid_score = validate(models, encoded_less_toxic, encoded_more_toxic)
    test_predictions = predict(models, encoded_test_data)
    submission = pd.DataFrame(
        {"comment_id": test_data.comment_id, "score": test_predictions}
    )
    submission.to_csv(args.pred_save_path, index=False)