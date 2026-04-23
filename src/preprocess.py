from __future__ import annotations

import re
from urllib.parse import urlparse

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
URL_HINT_PATTERN = re.compile(r"^(https?://|www\.)", re.IGNORECASE)


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    if not URL_HINT_PATTERN.search(url):
        url = f"http://{url}"
    return url


def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"http[s]?://\S+", " urltoken ", text)
    text = re.sub(r"[^a-z0-9@:/._'\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: str) -> list[str]:
    cleaned = clean_text(text)
    return [
        token
        for token in TOKEN_PATTERN.findall(cleaned)
        if token not in ENGLISH_STOP_WORDS and len(token) > 1
    ]


def preprocess_text(text: str) -> str:
    return " ".join(tokenize_text(text))


def detect_input_type(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "unknown"
    if URL_HINT_PATTERN.search(value) or (("." in value) and (" " not in value)):
        parsed = urlparse(normalize_url(value))
        if parsed.netloc:
            return "url"
    return "text"
