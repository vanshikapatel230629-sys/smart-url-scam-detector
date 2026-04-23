from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.feature_engineering import (
    SUSPICIOUS_TEXT_KEYWORDS,
    explain_url_flags,
    extract_url_features,
    highlight_suspicious_words,
    highlight_url,
)
from src.preprocess import clean_text, normalize_url, tokenize_text

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

RISK_LEVELS = {"safe": "Low", "suspicious": "Medium", "malicious": "High"}


def load_artifact(model_path: Path | None = None) -> dict[str, Any]:
    return joblib.load(model_path or MODEL_PATH)


def _top_probability(model, text_or_url: str) -> tuple[str, float, dict[str, float]]:
    probabilities = model.predict_proba([text_or_url])[0]
    label_scores = {
        label: round(float(score), 4) for label, score in zip(model.classes_, probabilities, strict=False)
    }
    top_label = max(label_scores, key=label_scores.get)
    return top_label, label_scores[top_label], label_scores


def _linear_text_contributions(model, text: str, predicted_label: str) -> list[str]:
    vectorizer = model.named_steps.get("vectorizer")
    if vectorizer is None:
        vectorizer = model.named_steps.get("features")
    classifier = model.named_steps.get("classifier")
    if vectorizer is None or classifier is None or not hasattr(classifier, "coef_"):
        return []

    vector = vectorizer.transform([text])
    class_index = list(classifier.classes_).index(predicted_label)
    row = classifier.coef_[class_index]
    contributions = vector.multiply(row).toarray()[0]
    if not np.any(contributions):
        return []
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_indices = contributions.argsort()[-5:][::-1]
    return [feature_names[index] for index in top_indices if contributions[index] > 0]


def analyze_url(url: str, artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    artifact = artifact or load_artifact()
    normalized = normalize_url(url)
    model = artifact["url_model"]["pipeline"]
    predicted_label, confidence, probabilities = _top_probability(model, normalized)
    features = extract_url_features(normalized)
    reasons = explain_url_flags(normalized)
    top_factors = sorted(features.items(), key=lambda item: item[1], reverse=True)[:5]
    return {
        "input_type": "url",
        "normalized_input": normalized,
        "prediction": predicted_label.title(),
        "confidence": round(confidence * 100, 2),
        "risk_level": RISK_LEVELS[predicted_label],
        "probabilities": probabilities,
        "explanation": reasons,
        "top_factors": [f"{name}: {value:.2f}" for name, value in top_factors if value > 0],
        "highlighted": highlight_url(normalized),
    }


def analyze_text(text: str, artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    artifact = artifact or load_artifact()
    cleaned = clean_text(text)
    model = artifact["text_model"]["pipeline"]
    predicted_label, confidence, probabilities = _top_probability(model, cleaned)

    tokens = tokenize_text(cleaned)
    keyword_hits = [token for token in tokens if token in SUSPICIOUS_TEXT_KEYWORDS]
    model_terms = _linear_text_contributions(model, cleaned, predicted_label)
    reasons = []
    if keyword_hits:
        reasons.append(f"Suspicious keywords detected: {', '.join(sorted(set(keyword_hits)))}.")
    if "urltoken" in cleaned or "http" in text.lower():
        reasons.append("Contains a URL or external link.")
    if any(word in cleaned for word in ["urgent", "final", "immediately", "warning"]):
        reasons.append("Uses pressure or urgency language.")
    if any(word in cleaned for word in ["otp", "password", "pin", "cvv"]):
        reasons.append("Requests sensitive credentials or payment details.")
    if model_terms:
        clean_terms = [term.split("__")[-1] for term in model_terms[:5]]
        reasons.append(f"Top contributing terms: {', '.join(clean_terms)}.")
    if not reasons:
        reasons.append("No strong scam indicators were found in the text.")

    return {
        "input_type": "text",
        "normalized_input": cleaned,
        "prediction": predicted_label.title(),
        "confidence": round(confidence * 100, 2),
        "risk_level": RISK_LEVELS[predicted_label],
        "probabilities": probabilities,
        "explanation": reasons,
        "top_factors": [f"keyword: {token}" for token in sorted(set(keyword_hits))]
        or [term.split("__")[-1] for term in model_terms[:5]],
        "highlighted": highlight_suspicious_words(text),
    }
