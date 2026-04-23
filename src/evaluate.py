from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.feature_engineering import TextSignalExtractor, URLFeatureExtractor
from src.preprocess import preprocess_text

LABEL_ORDER = ["safe", "suspicious", "malicious"]


def evaluate_predictions(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABEL_ORDER).tolist(),
    }


def rank_url_features(urls: pd.Series, labels: pd.Series) -> list[dict[str, float]]:
    extractor = URLFeatureExtractor()
    features = extractor.transform(urls)
    scores = mutual_info_classif(features, labels, discrete_features=False, random_state=42)
    ranking = (
        pd.DataFrame({"feature": features.columns, "importance": scores})
        .sort_values("importance", ascending=False)
        .head(10)
    )
    return ranking.to_dict(orient="records")


def rank_text_features(messages: pd.Series, labels: pd.Series) -> list[dict[str, float]]:
    features = FeatureUnion(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess_text,
                    tokenizer=str.split,
                    token_pattern=None,
                    ngram_range=(1, 2),
                    min_df=1,
                ),
            ),
            ("signals", TextSignalExtractor()),
        ]
    )
    matrix = features.fit_transform(messages)
    scores, _ = chi2(matrix, labels)
    ranking = (
        pd.DataFrame(
            {
                "feature": features.get_feature_names_out(),
                "importance": np.nan_to_num(scores),
            }
        )
        .sort_values("importance", ascending=False)
        .head(15)
    )
    return ranking.to_dict(orient="records")
