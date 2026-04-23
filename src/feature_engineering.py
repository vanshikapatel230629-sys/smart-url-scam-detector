from __future__ import annotations

from html import escape
import re
from typing import Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.preprocess import normalize_url

SUSPICIOUS_TLDS = {
    "ru",
    "tk",
    "xyz",
    "top",
    "gq",
    "ml",
    "ga",
    "cc",
    "cam",
    "live",
    "biz",
    "info",
}
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly"}
RISKY_URL_TOKENS = {
    "login",
    "verify",
    "update",
    "secure",
    "account",
    "bonus",
    "reward",
    "bank",
    "payment",
    "wallet",
    "otp",
    "signin",
    "confirm",
    "gift",
}
SUSPICIOUS_TEXT_KEYWORDS = {
    "urgent",
    "verify",
    "account",
    "password",
    "otp",
    "gift",
    "winner",
    "loan",
    "bank",
    "click",
    "link",
    "claim",
    "free",
    "bonus",
    "refund",
    "payment",
    "cvv",
    "pin",
    "aadhaar",
    "wallet",
}
URGENT_TERMS = {"urgent", "immediately", "final", "warning", "alert", "fast", "hurry", "now"}
CREDENTIAL_TERMS = {"otp", "password", "pin", "cvv", "card", "debit", "credit", "login"}
REWARD_TERMS = {"gift", "winner", "reward", "bonus", "prize", "cash", "loan", "refund"}
SECURITY_TERMS = {"verify", "account", "security", "confirm", "update", "bank", "wallet", "kyc"}


def extract_url_features(url: str) -> dict[str, float]:
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    query = parsed.query.lower()
    domain_parts = [part for part in host.split(".") if part]
    tld = domain_parts[-1] if domain_parts else ""
    digit_count = sum(char.isdigit() for char in normalized)

    return {
        "url_length": float(len(normalized)),
        "has_at_symbol": float("@" in normalized),
        "has_hyphen": float("-" in host),
        "double_slash_count": float(max(normalized.count("//") - 1, 0)),
        "subdomain_count": float(max(len(domain_parts) - 2, 0)),
        "uses_https": float(parsed.scheme == "https"),
        "digit_ratio": float(digit_count / max(len(normalized), 1)),
        "dot_count": float(normalized.count(".")),
        "question_count": float(normalized.count("?")),
        "ampersand_count": float(normalized.count("&")),
        "equals_count": float(normalized.count("=")),
        "path_length": float(len(path)),
        "query_length": float(len(query)),
        "host_length": float(len(host)),
        "suspicious_tld": float(tld in SUSPICIOUS_TLDS),
        "is_shortened": float(host in SHORTENERS),
        "risky_token_count": float(sum(token in normalized.lower() for token in RISKY_URL_TOKENS)),
    }


class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X: Iterable[str], y: Iterable[str] | None = None) -> "URLFeatureExtractor":
        return self

    def transform(self, X: Iterable[str]) -> pd.DataFrame:
        rows = [extract_url_features(url) for url in X]
        return pd.DataFrame(rows).fillna(0.0)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        return np.array(list(extract_url_features("https://example.com").keys()))


class TextSignalExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X: Iterable[str], y: Iterable[str] | None = None) -> "TextSignalExtractor":
        return self

    def transform(self, X: Iterable[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for text in X:
            normalized = (text or "").lower()
            tokens = re.findall(r"[a-z0-9']+", normalized)
            rows.append(
                [
                    float("http" in normalized or "www." in normalized or "link" in normalized),
                    float(sum(token in URGENT_TERMS for token in tokens)),
                    float(sum(token in CREDENTIAL_TERMS for token in tokens)),
                    float(sum(token in REWARD_TERMS for token in tokens)),
                    float(sum(token in SECURITY_TERMS for token in tokens)),
                    float(normalized.count("!")),
                    float(sum(char.isdigit() for char in normalized)),
                ]
            )
        return np.array(rows, dtype=float)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        return np.array(
            [
                "has_link",
                "urgent_term_count",
                "credential_term_count",
                "reward_term_count",
                "security_term_count",
                "exclamation_count",
                "digit_count",
            ]
        )


def explain_url_flags(url: str) -> list[str]:
    features = extract_url_features(url)
    reasons: list[str] = []
    if features["has_at_symbol"]:
        reasons.append("Contains '@', which can hide the real destination.")
    if features["has_hyphen"]:
        reasons.append("Uses a hyphenated domain that often appears in impersonation links.")
    if features["double_slash_count"] > 0:
        reasons.append("Contains an extra '//' pattern beyond the protocol.")
    if features["subdomain_count"] >= 2:
        reasons.append("Has many subdomains, which is common in deceptive URLs.")
    if not features["uses_https"]:
        reasons.append("Does not use HTTPS.")
    if features["suspicious_tld"]:
        reasons.append("Uses a high-risk top-level domain.")
    if features["is_shortened"]:
        reasons.append("Uses a shortened URL, which hides the final landing page.")
    if features["risky_token_count"] >= 2:
        reasons.append("Contains multiple high-risk words such as login, verify, or reward.")
    if features["digit_ratio"] > 0.08:
        reasons.append("Contains an unusual number of digits.")
    return reasons or ["No obvious risky URL patterns were detected."]


def highlight_url(url: str) -> str:
    highlighted = escape(url)
    for token in sorted(RISKY_URL_TOKENS, key=len, reverse=True):
        highlighted = highlighted.replace(token, f"<mark>{token}</mark>")
        highlighted = highlighted.replace(token.title(), f"<mark>{token.title()}</mark>")
        highlighted = highlighted.replace(token.upper(), f"<mark>{token.upper()}</mark>")
    for symbol in ["@", "//", "-", ".tk", ".ru", ".xyz", ".top", ".cc"]:
        highlighted = highlighted.replace(symbol, f"<mark>{escape(symbol)}</mark>")
    return highlighted


def highlight_suspicious_words(text: str) -> str:
    words = text.split()
    highlighted_words = []
    for word in words:
        normalized = "".join(char.lower() for char in word if char.isalnum())
        if normalized in SUSPICIOUS_TEXT_KEYWORDS or word.lower().startswith("http"):
            highlighted_words.append(f"<mark>{escape(word)}</mark>")
        else:
            highlighted_words.append(escape(word))
    return " ".join(highlighted_words)
