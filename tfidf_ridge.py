import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from typing import List
import joblib
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--valid_path", type=str, default="data/valid.csv")
    parser.add_argument("--test_path", type=str, default="data/comments_to_score.csv")
    parser.add_argument("--model_save_dir", type=str, default=".")
    parser.add_argument("--pred_save_path", type=str, default="./submission.csv")
    parser.add_argument("--ridge_alpha", type=float, default=None)
    parser.add_argument("--max_alpha", type=float, default=1.0)
    parser.add_argument("--tokenization_scheme", type=str, default="word")
    parser.add_argument("--min_df", type=int, default=0)
    parser.add_argument("--max_df", type=float, default=0.8),
    parser.add_argument("--ngram_min", type=int, default=1),
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--num_folds", type=int, default=5)
    return parser.parse_args()


def train(
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    encoder: TfidfVectorizer,
    alpha: float = None,
    max_alpha: float = 1.0,
):
    best_model = None
    if alpha is None:
        best_alpha = None
        best_score = 0
        for alpha in np.linspace(0.1, max_alpha, max_alpha * 10):
            regressor = Ridge(alpha=alpha)
            model = Pipeline([("tfidf", encoder), ("ridge", regressor)])
            model.fit(train_data.text, train_data.target)
            score = validate([model], valid_data)
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_model = model
            print(f"alpha: {alpha} | score: {score}")
    else:
        regressor = Ridge(alpha=alpha)
        best_model = Pipeline([("tfidf", encoder), ("ridge", regressor)])
        best_model.fit(train_data.text, train_data.target)
        best_score = validate([best_model], valid_data)
    print(f"final alpha: {best_alpha} | final score: {best_score}")


def train_fold(
    fold: int,
    train_data: pd.DataFrame,
    oof_data: pd.DataFrame,
    encoder: TfidfVectorizer,
    alpha: float = None,
    max_alpha: float = 1.0,
) -> Pipeline:
    min_mse = float("inf")
    best_model = None
    best_alpha = alpha
    if alpha is None:  # if no alpha is supplied then tune it
        print(f"Tuning alpha for fold {fold}...")
        for alpha in np.linspace(0.1, max_alpha, max_alpha * 10):
            regressor = Ridge(alpha=alpha)
            model = Pipeline([("tfidf", encoder), ("ridge", regressor)])
            model.fit(train_data.text, train_data.target)
            predictions = model.predict(oof_data.text)
            mse = mean_squared_error(oof_data.target, predictions)
            print(f"fold: {fold} | alpha: {alpha} | mse: {mse}")
            if mse < min_mse:
                min_mse = mse
                best_model = model
                best_alpha = alpha
    else:  # use supplied alpha to fit the regressor
        print(f"Fitting fold {fold} using alpha={best_alpha}...")
        regressor = Ridge(alpha=best_alpha)
        best_model = Pipeline([("tfidf", encoder), ("ridge", regressor)])
        best_model.fit(train_data.text, train_data.target)
        predictions = best_model.predict(oof_data.text)
        min_mse = mean_squared_error(oof_data.target, predictions)
    print(f"best model | alpha: {best_alpha} | mse: {min_mse}\n")
    return best_model, min_mse


def get_encoder(
    tokenization_scheme: str,
    min_df: int,
    max_df: float,
    ngram_min: int,
    ngram_max: int,
) -> TfidfVectorizer:
    encoder = TfidfVectorizer(
        analyzer=tokenization_scheme,
        min_df=min_df,
        max_df=max_df,
        ngram_range=(ngram_min, ngram_max),
    )
    return encoder


def make_folds(data: pd.DataFrame, num_folds: int, seed: int) -> pd.DataFrame:
    skf = StratifiedKFold(shuffle=True, random_state=seed)
    stratified_targets = pd.cut(data.target, num_folds, labels=False)
    data["fold"] = -1
    for fold, (_, valid_index) in enumerate(skf.split(data.text, stratified_targets)):
        data.loc[valid_index, "fold"] = fold
    return data


def validate(models: List[Pipeline], test_data: pd.DataFrame) -> float:
    less_toxic_scores = []
    more_toxic_scores = []
    for model in models:
        less_toxic_scores.append(model.predict(test_data.less_toxic))
        more_toxic_scores.append(model.predict(test_data.more_toxic))
    mean_less_toxic = np.mean(less_toxic_scores, axis=0)
    mean_more_toxic = np.mean(more_toxic_scores, axis=0)
    return sum(mean_less_toxic < mean_more_toxic) / len(test_data)


def predict(models: List[Pipeline], test_data: pd.DataFrame) -> np.ndarray:
    print("Generating predictions...")
    fold_predictions = []
    for model in models:
        predictions = model.predict(test_data.text)
        fold_predictions.append(predictions)
    return np.mean(fold_predictions, axis=0)


def save_model(model: Pipeline, fold: int, save_dir: str) -> None:
    save_path = os.path.join(save_dir, f"tfidf_ridge_fold_{fold}.pkl")
    joblib.dump(model, save_path)


if __name__ == "__main__":
    args = parse_args()
    data = pd.read_csv(args.train_path)
    valid_data = pd.read_csv(args.valid_path)
    test_data = pd.read_csv(args.test_path)
    models = []
    mse_scores = []
    encoder = get_encoder(
        args.tokenization_scheme,
        args.min_df,
        args.max_df,
        args.ngram_min,
        args.ngram_max,
    )
    if args.num_folds > 1:
        if "fold" not in data.columns:
            data = make_folds(data, args.num_folds, args.seed)
        for fold in range(args.num_folds):
            train_data = data[data.fold != fold]
            oof_data = data[data.fold == fold]
            model, mse = train_fold(
                fold,
                train_data,
                oof_data,
                args.tokenization_scheme,
                args.min_df,
                args.max_df,
                args.ngram_min,
                args.ngram_max,
                args.ridge_alpha,
                args.max_alpha,
            )
            mse_scores.append(mse)
            models.append(model)
            save_model(model, fold, args.model_save_dir)
        print(f"cv mse: {np.mean(mse_scores)}")
        valid_score = validate(models, valid_data)
        print(f"valid score (mean of {args.num_folds} folds): {valid_score}")
    else:
        alpha = args.ridge_alpha
        if alpha is None:
            print("Defaulting to alpha=1")
            alpha = 1
        model = train(data, valid_data, encoder, alpha, args.max_alpha)
        models.append(model)
    print("Generating test set predictions...")
    predictions = predict(models, test_data)
    submission = pd.DataFrame(
        {"comment_id": test_data.comment_id, "score": predictions}
    )
    submission.to_csv(args.pred_save_path, index=False)
    print(f"Saved predictions to {args.pred_save_path}")