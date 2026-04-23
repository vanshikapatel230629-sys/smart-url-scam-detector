from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from src.evaluate import LABEL_ORDER, evaluate_predictions, rank_text_features, rank_url_features
from src.feature_engineering import TextSignalExtractor, URLFeatureExtractor
from src.preprocess import preprocess_text

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    urls = pd.read_csv(DATA_DIR / "urls.csv")
    urls["label"] = urls["label"].replace({"natural": "safe"})
    messages = pd.read_csv(DATA_DIR / "messages.csv")
    return urls.dropna(), messages.dropna()


def build_url_candidates() -> dict[str, Pipeline]:
    base_steps = [
        ("features", URLFeatureExtractor()),
        ("scaler", MinMaxScaler()),
    ]
    return {
        "Logistic Regression": Pipeline(
            base_steps
            + [("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))]
        ),
        "Random Forest": Pipeline(
            [
                ("features", URLFeatureExtractor()),
                ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
            ]
        ),
        "Naive Bayes": Pipeline(base_steps + [("classifier", GaussianNB())]),
        "SVM": Pipeline(
            base_steps
            + [("classifier", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42))]
        ),
    }


def build_text_candidates() -> dict[str, Pipeline]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    feature_stack = FeatureUnion(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess_text,
                    tokenizer=str.split,
                    token_pattern=None,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            ("signals", TextSignalExtractor()),
        ]
    )
    return {
        "Logistic Regression": Pipeline(
            [
                ("features", feature_stack),
                ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("features", feature_stack),
                ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
            ]
        ),
        "Naive Bayes": Pipeline(
            [
                ("features", feature_stack),
                ("classifier", MultinomialNB()),
            ]
        ),
        "SVM": Pipeline(
            [
                ("features", feature_stack),
                ("classifier", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)),
            ]
        ),
    }


def train_and_select(X: pd.Series, y: pd.Series, candidates: dict[str, Pipeline]) -> dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    best_name = ""
    best_model: Pipeline | None = None
    best_metrics: dict[str, Any] | None = None
    all_results: dict[str, Any] = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = evaluate_predictions(y_test.tolist(), predictions.tolist())
        all_results[name] = metrics
        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_name = name
            best_model = model
            best_metrics = metrics

    assert best_model is not None and best_metrics is not None
    return {
        "best_model_name": best_name,
        "best_model": best_model,
        "best_metrics": best_metrics,
        "all_results": all_results,
        "test_samples": X_test.tolist(),
        "test_labels": y_test.tolist(),
    }


def train_all() -> dict[str, Any]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    urls, messages = load_datasets()

    url_result = train_and_select(urls["url"], urls["label"], build_url_candidates())
    text_result = train_and_select(messages["message"], messages["label"], build_text_candidates())

    artifact = {
        "project_name": "Smart URL & Scam Detector",
        "labels": LABEL_ORDER,
        "url_model": {
            "name": url_result["best_model_name"],
            "pipeline": url_result["best_model"],
            "metrics": url_result["best_metrics"],
            "candidates": url_result["all_results"],
            "feature_importance": rank_url_features(urls["url"], urls["label"]),
            "distribution": urls["label"].value_counts().sort_index().to_dict(),
        },
        "text_model": {
            "name": text_result["best_model_name"],
            "pipeline": text_result["best_model"],
            "metrics": text_result["best_metrics"],
            "candidates": text_result["all_results"],
            "feature_importance": rank_text_features(messages["message"], messages["label"]),
            "distribution": messages["label"].value_counts().sort_index().to_dict(),
        },
        "examples": {
            "safe_url": "https://www.google.com/search?q=cybersecurity",
            "malicious_url": "http://secure-login@verify-bank.ru/reset",
            "safe_message": "Your monthly subscription payment was received successfully.",
            "malicious_message": "Urgent! Verify your bank account and share OTP to avoid suspension.",
        },
    }

    joblib.dump(artifact, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(_make_json_safe(artifact), indent=2), encoding="utf-8")
    return artifact


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _make_json_safe(val) for key, val in value.items() if key != "pipeline"}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    trained = train_all()
    print(f"Saved trained artifact to: {MODEL_PATH}")
    print(f"Best URL model: {trained['url_model']['name']}")
    print(f"Best Text model: {trained['text_model']['name']}")
