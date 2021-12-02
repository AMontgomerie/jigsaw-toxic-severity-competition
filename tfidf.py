import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="data/train.csv")
    parser.add_argument("--test_data_path", type=str, default="data/valid.csv")
    return parser.parse_args()


def train(train_data: pd.DataFrame) -> Pipeline:
    model = Pipeline([("tfidf", TfidfVectorizer()), ("ridge", Ridge())])
    model.fit(train_data.text, train_data.target)
    return model


def test(models: List[Pipeline], test_data: pd.DataFrame) -> float:
    less_toxic_scores = []
    more_toxic_scores = []
    for model in models:
        less_toxic_scores.append(model.predict(test_data.less_toxic))
        more_toxic_scores.append(model.predict(test_data.more_toxic))
    mean_less_toxic = np.mean(less_toxic_scores, axis=0)
    mean_more_toxic = np.mean(more_toxic_scores, axis=0)
    return sum(mean_less_toxic < mean_more_toxic) / len(test_data)


if __name__ == "__main__":
    args = parse_args()
    data = pd.read_csv(args.train_data_path)
    test_data = pd.read_csv(args.test_data_path)
    models = []
    valid_scores = []
    for fold in range(5):
        train_data = data[data.fold != fold]
        valid_data = data[data.fold == fold]
        model = train(train_data)
        preds = model.predict(valid_data.text)
        mse = mean_squared_error(valid_data.target, preds)
        models.append(model)
        valid_scores.append(mse)
    print(f"validation MSE by fold: {valid_scores}")
    test_score = test(models, test_data)
    print(test_score)