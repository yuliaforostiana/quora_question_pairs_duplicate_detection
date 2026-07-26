import time
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances

# Configuration

MODEL_PATH = "models"
PROCESS_DATA_PATH = "preprocess_data"

XGB_MODEL_FILE = f"{MODEL_PATH}/xgboost_tfidf_se1.joblib"
EMBEDDING_NAME_FILE = f"{PROCESS_DATA_PATH}/embedding_model_name.txt"


embedding_model = None
xgb_model = None

# Preprocessing

def minimal_clean(text: str) -> str:
    """Strips extra whitespace. Nothing else -- the embedding model
    is trained on plain, natural text."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def build_features(question1: str, question2: str) -> np.ndarray:
    """
    Turns a pair of questions into a numeric feature vector for XGBoost.

    Steps:
    1. Clean the text
    2. Turn each question into an embedding (a vector of numbers)
    3. Compute the cosine and euclidean distance between the embeddings
    4. Concatenate everything into one vector: [emb1, emb2, cosine, euclidean]
    """
    clean1 = minimal_clean(question1)
    clean2 = minimal_clean(question2)

    # encode() expects a list of strings, so we pass a single-item list
    emb1 = embedding_model.encode([clean1])[0]
    emb2 = embedding_model.encode([clean2])[0]

    # paired_..._distances also expect arrays of several elements,
    # so we wrap the embeddings in another list
    cosine = 1 - paired_cosine_distances([emb1], [emb2])[0]
    euclidean = paired_euclidean_distances([emb1], [emb2])[0]

    # Concatenate everything into one flat feature vector
    features = np.concatenate([emb1, emb2, [cosine, euclidean]])
    return features.reshape(1, -1)  # XGBoost expects a 2D array (even for 1 example)


def load_models():
    """Loads both models once, when the service starts."""
    global embedding_model, xgb_model

    with open(EMBEDDING_NAME_FILE, "r") as f:
        embedding_model_name = f.read().strip()

    print(f"Loading sentence-transformer: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)

    print(f"Loading XGBoost model: {XGB_MODEL_FILE}")
    xgb_model = joblib.load(XGB_MODEL_FILE)

    print("Models loaded, service is ready!")


# Request / Response schemes

class PredictRequest(BaseModel):
    question1: str
    question2: str


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    label: str
    inference_time_ms: float


# Fast API

app = FastAPI(title="Quora Duplicate Questions API")


@app.on_event("startup")
def on_startup():
    """This function runs once, when the server starts up."""
    load_models()


@app.get("/health")
def health():
    """A simple endpoint to check that the service is alive and models are loaded."""
    return {
        "status": "ok" if xgb_model is not None else "not_ready",
        "models_loaded": xgb_model is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="Models are not loaded yet")

    start = time.perf_counter()

    X = build_features(request.question1, request.question2)

    prediction = int(xgb_model.predict(X)[0])
    probability = float(xgb_model.predict_proba(X)[0][1])
    label = "Duplicate" if prediction == 1 else "Not duplicate"

    elapsed_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        prediction=prediction,
        probability=probability,
        label=label,
        inference_time_ms=elapsed_ms,
    )