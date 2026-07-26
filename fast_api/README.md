# Quora Duplicate Questions API

A FastAPI service that predicts whether two questions are duplicates, powered by an XGBoost model trained on sentence embeddings (cosine + euclidean distance features).

## Overview

This service reproduces the inference pipeline from the training notebook:

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
├── app.py           # FastAPI service
├── requirements.txt
├── Dockerfile
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

## Running the API

```bash
uvicorn app_simple:app --host 0.0.0.0 --port 8000
```

Then open **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

> Note: navigating to `http://0.0.0.0:8000` directly in a browser will not work — `0.0.0.0` means "listen on all network interfaces," it isn't a browsable address. Use `127.0.0.1` or `localhost` instead.

## Endpoints

| Method | Path      | Description                          |
|--------|-----------|---------------------------------------|
| GET    | `/health` | Check that the models are loaded      |
| POST   | `/predict`| Predict whether two questions are duplicates |

**Example request:**
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "question1": "How do I learn Python?",
    "question2": "What is the best way to learn Python programming?"
  }'
```

**Example response:**
```json
{
  "prediction": 1,
  "probability": 0.87,
  "label": "Duplicate",
  "inference_time_ms": 12.4
}
```

## Running with Docker

```bash
docker build -t fast_api .
docker run -p 8000:8000 quora-duplicate-api
```

Make sure `models/` and `preprocess_data/` contain the required files before building the image, since `Dockerfile` copies them into the container.

## Implementation notes

- **Models are loaded once at startup** (via FastAPI's `startup` event), not on every request — loading a sentence-transformer on each call would add seconds of latency.
- **`minimal_clean`** only normalizes whitespace, deliberately skipping lowercasing/lemmatization, since the embedding model is trained on natural, unmodified text.
- **Feature order** must match training exactly: `[emb_q1, emb_q2, cosine, euclidean]`. If the deployed model was trained on a different feature order or set, predictions will be invalid — verify this before using in production.
- The service runs with a single worker (`--workers 1`) by design; the sentence-transformer and XGBoost model are memory-heavy, so scaling is intended to be done by running multiple containers rather than multiple workers inside one.

## Next steps for production

- Add authentication (API key or OAuth)
- Add rate limiting
- Add request/response logging and metrics (e.g. Prometheus)
- Add `max_length` validation on input text
- Consider ONNX export / quantization if latency becomes a bottleneck at scale
