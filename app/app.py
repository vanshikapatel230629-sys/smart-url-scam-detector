from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import analyze_text, analyze_url, load_artifact
from src.train_model import MODEL_PATH, train_all

st.set_page_config(
    page_title="Smart URL & Scam Detector",
    layout="wide",
)

LABEL_COLORS = {"Safe": "#16a34a", "Suspicious": "#f59e0b", "Malicious": "#dc2626"}


@st.cache_resource
def get_artifact():
    if not MODEL_PATH.exists():
        return train_all()
    return load_artifact()


def render_result_card(result: dict):
    color = LABEL_COLORS.get(result["prediction"], "#2563eb")
    st.markdown(
        f"""
        <div style="padding: 1rem 1.2rem; border-radius: 14px; background: {color}14; border-left: 8px solid {color};">
            <h3 style="margin: 0; color: {color};">{result['prediction']} ({result['risk_level']} Risk)</h3>
            <p style="margin: 0.5rem 0 0 0; font-size: 1rem;">Confidence: <strong>{result['confidence']}%</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_chart(probabilities: dict[str, float], title: str):
    df = pd.DataFrame({"Label": list(probabilities.keys()), "Probability": list(probabilities.values())})
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(data=df, x="Label", y="Probability", palette=["#16a34a", "#f59e0b", "#dc2626"], ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("Probability")
    st.pyplot(fig, clear_figure=True)


def render_confusion_matrix(matrix: list[list[int]], title: str):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=["safe", "suspicious", "malicious"],
        yticklabels=["safe", "suspicious", "malicious"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig, clear_figure=True)


def render_distribution(distribution: dict[str, int], title: str):
    df = pd.DataFrame({"Label": list(distribution.keys()), "Count": list(distribution.values())})
    fig, ax = plt.subplots(figsize=(5, 3.2))
    sns.barplot(data=df, x="Label", y="Count", palette=["#16a34a", "#f59e0b", "#dc2626"], ax=ax)
    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)


def render_feature_importance(records: list[dict[str, float]], title: str):
    df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(7, max(3.5, len(df) * 0.35)))
    sns.barplot(data=df.sort_values("importance"), x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance Score")
    st.pyplot(fig, clear_figure=True)


artifact = get_artifact()

st.title("Smart URL & Scam Detector")
st.caption("A lightweight cybersecurity assistant for detecting phishing URLs and scam messages.")

with st.sidebar:
    st.subheader("Model Summary")
    st.write(f"URL model: **{artifact['url_model']['name']}**")
    st.write(f"Text model: **{artifact['text_model']['name']}**")
    if st.button("Retrain Models"):
        artifact = train_all()
        st.success("Models retrained and saved successfully.")

tab_url, tab_text, tab_dashboard = st.tabs(["URL Scanner", "Message Scanner", "Analytics Dashboard"])

with tab_url:
    st.subheader("Analyze a URL")
    url_input = st.text_input(
        "Enter a website or login URL",
        value=artifact["examples"]["malicious_url"],
        placeholder="https://example.com",
    )
    if st.button("Analyze URL", type="primary"):
        result = analyze_url(url_input, artifact)
        render_result_card(result)
        col1, col2 = st.columns([1, 1])
        with col1:
            render_probability_chart(result["probabilities"], "URL Risk Probabilities")
        with col2:
            st.markdown("**Why it was flagged**")
            for item in result["explanation"]:
                st.write(f"- {item}")
            if result["top_factors"]:
                st.markdown("**Top contributing factors**")
                for item in result["top_factors"]:
                    st.write(f"- {item}")
        st.markdown("**Risky URL parts highlighted**")
        st.markdown(f"<div style='font-size:1rem'>{result['highlighted']}</div>", unsafe_allow_html=True)

with tab_text:
    st.subheader("Analyze a Message")
    text_input = st.text_area(
        "Paste an SMS, email, or chat message",
        value=artifact["examples"]["malicious_message"],
        height=180,
    )
    if st.button("Analyze Message", type="primary"):
        result = analyze_text(text_input, artifact)
        render_result_card(result)
        col1, col2 = st.columns([1, 1])
        with col1:
            render_probability_chart(result["probabilities"], "Message Risk Probabilities")
        with col2:
            st.markdown("**Why it was flagged**")
            for item in result["explanation"]:
                st.write(f"- {item}")
            if result["top_factors"]:
                st.markdown("**Suspicious signals**")
                for item in result["top_factors"]:
                    st.write(f"- {item}")
        st.markdown("**Suspicious words highlighted**")
        st.markdown(f"<div style='font-size:1rem'>{result['highlighted']}</div>", unsafe_allow_html=True)

with tab_dashboard:
    st.subheader("Training Insights")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("URL Accuracy", artifact["url_model"]["metrics"]["accuracy"])
    metric_col2.metric("URL F1", artifact["url_model"]["metrics"]["macro_f1"])
    metric_col3.metric("Text Accuracy", artifact["text_model"]["metrics"]["accuracy"])
    metric_col4.metric("Text F1", artifact["text_model"]["metrics"]["macro_f1"])

    left, right = st.columns(2)
    with left:
        render_distribution(artifact["url_model"]["distribution"], "URL Label Distribution")
        render_confusion_matrix(artifact["url_model"]["metrics"]["confusion_matrix"], "URL Confusion Matrix")
    with right:
        render_distribution(artifact["text_model"]["distribution"], "Message Label Distribution")
        render_confusion_matrix(artifact["text_model"]["metrics"]["confusion_matrix"], "Message Confusion Matrix")

    left, right = st.columns(2)
    with left:
        render_feature_importance(artifact["url_model"]["feature_importance"], "Top URL Feature Importance")
    with right:
        render_feature_importance(artifact["text_model"]["feature_importance"], "Top Text Feature Importance")
