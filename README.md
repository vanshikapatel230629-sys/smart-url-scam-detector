# Smart URL & Scam Detector

Smart URL & Scam Detector is an end-to-end machine learning project that detects malicious URLs and scam messages. It classifies each input as **Safe**, **Suspicious**, or **Malicious**, shows a confidence score, and explains why the content was flagged.

## Highlights

- Supervised learning for URL and message classification
- URL feature engineering:
  - URL length
  - `@`, `-`, and extra `//`
  - subdomain count
  - HTTPS usage
  - domain and structure signals
- Text preprocessing:
  - tokenization
  - stopword removal
  - TF-IDF vectorization
  - handcrafted scam-signal features
- Model comparison:
  - Logistic Regression
  - Random Forest
  - Naive Bayes
  - SVM
- Automatic best-model selection using macro F1-score
- Explanation engine with suspicious word and URL highlighting
- Streamlit dashboard with metrics, confusion matrices, distributions, and feature importance
- GitHub-ready structure with CI, runtime config, and deployment files

## Project Structure

```text
malicious-detector/
├── .github/workflows/ci.yml
├── .streamlit/config.toml
├── app/
│   └── app.py
├── data/
│   ├── messages.csv
│   └── urls.csv
├── models/
│   ├── metrics.json
│   └── model.pkl
├── src/
│   ├── __init__.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train_model.py
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── runtime.txt
```

## Local Setup

```bash
git clone <your-repo-url>
cd malicious-detector
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train_model
python -m streamlit run app/app.py
```

## How It Works

### URL pipeline

- Extracts structural features from each URL
- Trains and compares four classifiers
- Selects the best model automatically
- Explains risky patterns like `@`, suspicious TLDs, missing HTTPS, and deceptive subdomains

### Message pipeline

- Cleans text and removes stopwords
- Builds TF-IDF features plus scam-specific signals
- Trains and compares four classifiers
- Highlights suspicious terms such as `otp`, `verify`, `bank`, and `urgent`

## Current Model Snapshot

The included trained artifact is ready to use immediately from `models/model.pkl`.

- URL model: `Random Forest`
- URL accuracy: `0.9167`
- URL macro F1: `0.8963`
- Text model: `SVM`
- Text accuracy: `0.8182`
- Text macro F1: `0.8056`

## Example Inputs

### URL examples

- Safe: `https://www.google.com/search?q=cybersecurity`
- Suspicious: `https://secure-account-check.info/verify`
- Malicious: `http://secure-login@verify-bank.ru/reset`

### Message examples

- Safe: `Your monthly subscription payment was received successfully.`
- Suspicious: `Final notice: verify your wallet account for uninterrupted access.`
- Malicious: `Urgent! Verify your bank account and share OTP to avoid suspension.`

## Example Output

```text
Prediction: Malicious
Risk Level: High
Confidence: 61.20%
Why flagged:
- Contains '@', which can hide the real destination.
- Uses a hyphenated domain that often appears in impersonation links.
- Does not use HTTPS.
- Uses a high-risk top-level domain.
```

## Push To GitHub

```bash
cd malicious-detector
git init
git add .
git commit -m "Initial commit: Smart URL & Scam Detector"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

## Deploy To Streamlit Community Cloud

1. Push this project to a public GitHub repository.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**.
4. Select your GitHub repository.
5. Set the main file path to `app/app.py`.
6. Deploy the app.

The app is cloud-ready because:

- dependencies are listed in `requirements.txt`
- Python version is defined in `runtime.txt`
- Streamlit theme and server config are defined in `.streamlit/config.toml`
- the app auto-loads the saved model and retrains only if the model file is missing

## Notes

- The bundled datasets are starter datasets for demonstration and portfolio use.
- You can replace the CSV files with larger real-world datasets without changing the app structure.
- Predictions are lightweight, CPU-friendly, and designed for quick response.
