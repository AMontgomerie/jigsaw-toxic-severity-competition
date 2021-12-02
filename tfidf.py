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
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--valid_path", type=str, default="data/valid.csv")
    parser.add_argument("--test_path", type=str, default="data/comments_to_score.csv")
    parser.add_argument("--save_path", type=str, default="./submission.csv")
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


def predict(
    models: List[Pipeline], test_data: pd.DataFrame, submission_path: str
) -> None:
    print("Generating predictions...")
    fold_predictions = []
    for model in models:
        predictions = model.predict(test_data.text)
        fold_predictions.append(predictions)
    mean_predictions = np.mean(fold_predictions, axis=0)
    submission = pd.DataFrame(
        {"comment_id": test_data.comment_id, "score": mean_predictions}
    )
    submission.to_csv(submission_path, index=False)
    print(f"Saved predictions to {submission_path}")


if __name__ == "__main__":
    args = parse_args()
    data = pd.read_csv(args.train_data_path)
    valid_data = pd.read_csv(args.test_data_path)
    test_data = pd.read_csv(args.test_data_path)
    models = []
    oof_scores = []
    for fold in range(5):
        train_data = data[data.fold != fold]
        oof_data = data[data.fold == fold]
        model = train(train_data)
        predictions = model.predict(oof_data.text)
        mse = mean_squared_error(oof_data.target, predictions)
        models.append(model)
        oof_scores.append(mse)
    print(f"validation MSE by fold: {oof_scores}")
    test_score = test(models, valid_data)
    print(f"ranking test score (mean of 5 folds): {test_score}")
    predict(models, test_data, args.save_path)