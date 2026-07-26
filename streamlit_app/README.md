# Quora Duplicate Questions Checker (Streamlit)

A Streamlit app that predicts whether two questions are duplicates, powered by an XGBoost model trained on sentence embeddings (cosine + euclidean distance features). The app loads the models directly — no separate backend service required.


## [Live Demo](https://quoraquestionpairsduplicatedetection.streamlit.app/)

Interact with the deployed Streamlit application by entering two questions and receiving a real-time duplicate prediction with the corresponding confidence score.



## Overview

The app reproduces the inference pipeline from the training notebook:

```
question1, question2
      ↓
minimal_clean (whitespace normalization)
      ↓
SentenceTransformer.encode()  →  emb_q1, emb_q2
      ↓
cosine distance + euclidean distance
      ↓
feature vector = [emb_q1, emb_q2, cosine, euclidean]
      ↓
XGBoost.predict_proba()
      ↓
prediction + probability
```

## Project structure

```
.
├── app.py        # Streamlit app
├── requirements.txt
├── models/
│   └── xgboost_tfidf_se1.joblib      # not included, see Setup
└── preprocess_data/
    └── embedding_model_name.txt      # not included, see Setup
```

## Setup

1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add the required model files (not included in this repo):
   - `models/xgboost_tfidf_se1.joblib` — the trained XGBoost model
   - `preprocess_data/embedding_model_name.txt` — a text file containing the sentence-transformers model name used during training, e.g.:
     ```
     sentence-transformers/all-MiniLM-L6-v2
     ```

   Both folders must sit next to `streamlit_app.py` — the app reads them using relative paths (`models/...`, `preprocess_data/...`), so it must be run from the repository root.

## Running the app

```bash
streamlit run app.py
```

This opens a browser window at `http://localhost:8501` with a form for entering two questions and viewing the prediction.

Models are loaded once via `@st.cache_resource`, so the (slower) first run downloads/loads the sentence-transformer and XGBoost model, and every prediction after that reuses them from memory.

## Implementation notes

- **`minimal_clean`** only normalizes whitespace, deliberately skipping lowercasing/lemmatization, since the embedding model is trained on natural, unmodified text.
- **Feature order** must match training exactly: `[emb_q1, emb_q2, cosine, euclidean]`. If the deployed model was trained on a different feature order or set, predictions will be invalid — verify this before relying on the app.
- `@st.cache_resource` (rather than `@st.cache_data`) is used for model loading because it caches the actual model objects in memory across reruns/sessions, instead of trying to serialize them.

## Deploying to Streamlit Community Cloud

- Make sure `models/` and `preprocess_data/` are committed to the repository (or fetched at startup from external storage if they're too large for Git).
- `sentence-transformers` + `torch` are large dependencies; if the free tier's memory or build-time limits are hit, consider:
  - using a smaller sentence-transformers model,
  - installing a CPU-only `torch` wheel to reduce install size.

## Next steps

- Add input length validation
- Add a batch mode (upload a CSV of question pairs)
- Cache predictions for repeated question pairs
