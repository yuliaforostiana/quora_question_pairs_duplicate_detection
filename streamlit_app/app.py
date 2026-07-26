import time

import numpy as np
import joblib
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances

st.set_page_config(page_title="Quora Duplicate Checker", page_icon="🔍", layout="centered")


# Custom styling: soft, near-white blue background + card-like containers
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f9fc;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 0.5rem 0.5rem 0.5rem 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


MODEL_PATH = "models"
PROCESS_DATA_PATH = "preprocess_data"

XGB_MODEL_FILE = f"{MODEL_PATH}/xgboost_tfidf_se1.joblib"
EMBEDDING_NAME_FILE = f"{PROCESS_DATA_PATH}/embedding_model_name.txt"


# Load model and embedding model (once, cached)
@st.cache_resource
def load_model_bundle():
    with open(EMBEDDING_NAME_FILE, "r") as f:
        embedding_model_name = f.read().strip()

    embedding_model = SentenceTransformer(embedding_model_name)
    xgb_model = joblib.load(XGB_MODEL_FILE)

    return embedding_model, xgb_model


embedding_model, xgb_model = load_model_bundle()


# Preprocess
def minimal_clean(text: str) -> str:
    """Strips extra whitespace. Nothing else -- the embedding model
    is trained on plain, natural text."""
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def build_features(question1: str, question2: str) -> np.ndarray:
    """
    Turns a pair of questions into a numeric feature vector for XGBoost:
    [emb_q1, emb_q2, cosine_similarity, euclidean_distance]
    """
    clean1 = minimal_clean(question1)
    clean2 = minimal_clean(question2)

    emb1 = embedding_model.encode([clean1])[0]
    emb2 = embedding_model.encode([clean2])[0]

    cosine = 1 - paired_cosine_distances([emb1], [emb2])[0]
    euclidean = paired_euclidean_distances([emb1], [emb2])[0]

    features = np.concatenate([emb1, emb2, [cosine, euclidean]])
    return features.reshape(1, -1)


def predict(question1: str, question2: str):
    start = time.perf_counter()

    X = build_features(question1, question2)
    prediction = int(xgb_model.predict(X)[0])
    probability = float(xgb_model.predict_proba(X)[0][1])

    elapsed_ms = (time.perf_counter() - start) * 1000
    return prediction, probability, elapsed_ms


# Interface
st.title("🔍 Duplicate Question Checker")
st.markdown(
    "Enter two questions, and an XGBoost model (built on sentence "
    "embeddings) will predict whether they are duplicates."
)

with st.container(border=True):
    st.subheader("📝 Questions")
    question1 = st.text_area("Question 1", placeholder="e.g. How do I learn Python?")
    question2 = st.text_area("Question 2", placeholder="e.g. What is the best way to learn Python?")


# Prediction
st.divider()

if st.button("🔮 Check", type="primary", use_container_width=True):
    if not question1.strip() or not question2.strip():
        st.warning("Please fill in both fields.")
    else:
        with st.spinner("Analyzing..."):
            prediction, probability, elapsed_ms = predict(question1, question2)

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(f"✅ These are duplicates! (probability: {probability:.1%})")
        else:
            st.info(f"❌ These are different questions (duplicate probability: {probability:.1%})")

        st.progress(probability)
        st.caption(f"Inference time: {elapsed_ms:.1f} ms")