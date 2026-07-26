import numpy as np
import pandas as pd
import time
import torch
from datasets import Dataset

from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances

def clean_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows with a missing question1/question2."""
    before = len(df)
    df = df.dropna(subset=["question1", "question2"]).copy()
    print(f"Deleted {before - len(df)} rows")
    return df


def fix_identical_text_mislabels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes the label to 1 for pairs where question1 == question2 (as text),
    but is_duplicate == 0.
    """
    q1_clean = df["question1"].str.strip().str.lower()
    q2_clean = df["question2"].str.strip().str.lower()
    is_identical = q1_clean == q2_clean

    mislabeled = is_identical & (df["is_duplicate"] == 0)
    n_fixed = mislabeled.sum()

    df = df.copy()
    df.loc[mislabeled, "is_duplicate"] = 1
    print(f"Label fixed for {n_fixed} examples")
    return df


def dedupe_mirror_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles mirror pairs (qid1, qid2) and (qid2, qid1):
    -- consistent labels -> keep one row,
    -- conflicting labels -> drop both rows.
    """
    df = df.copy()
    df["_smaller_qid"] = np.minimum(df["qid1"], df["qid2"])
    df["_bigger_qid"] = np.maximum(df["qid1"], df["qid2"])

    group_key = ["_smaller_qid", "_bigger_qid"]
    label_nunique = df.groupby(group_key)["is_duplicate"].transform("nunique")

    conflicting_mask = label_nunique > 1
    n_conflicting_rows = conflicting_mask.sum()
    df = df[~conflicting_mask].copy()

    before = len(df)
    df = df.drop_duplicates(subset=group_key, keep="first")
    n_deduped = before - len(df)

    df = df.drop(columns=["_smaller_qid", "_bigger_qid"])
    print(f"Deleted {n_conflicting_rows} rows due to label conflict, "
          f"{n_deduped} rows due to duplication")
    return df



def df_cleaning(raw_df):
    df = clean_missing(raw_df)
    df = fix_identical_text_mislabels(df)
    df = dedupe_mirror_pairs(df)

    return df



def minimal_clean(text: str) -> str:
    """
    Minimal cleaning for sentence-transformers/BERT -- only strips extra
    whitespace. No lowercase/contractions/lemmatize/stop words: these
    models are trained on natural text, and aggressive normalization
    moves the input away from the distribution they were pretrained on.
    """
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def add_minimal_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["question1_clean"] = df["question1"].apply(minimal_clean)
    df["question2_clean"] = df["question2"].apply(minimal_clean)
    return df


def encode_questions(df: pd.DataFrame, model):
    """Encodes question1_clean/question2_clean into vectors. No fitting -- the model is already trained."""
    q1 = model.encode(df["question1_clean"].tolist(), show_progress_bar=True, batch_size=256)
    q2 = model.encode(df["question2_clean"].tolist(), show_progress_bar=True, batch_size=256)
    return q1, q2

def add_embedding_distance_features(df: pd.DataFrame, q1_emb, q2_emb) -> pd.DataFrame:
    df = df.copy()
    df["embedding_cosine"] = 1 - paired_cosine_distances(q1_emb, q2_emb)
    df["embedding_euclidean"] = paired_euclidean_distances(q1_emb, q2_emb)
    return df


def prepare_test_embeddings(
    test_df,
    embedding_model
):
    """
    Creates features for embedding-based models.

    Parameters
    ----------
    test_df : pd.DataFrame

    embedding_model : SentenceTransformer

    Returns
    -------
    pd.DataFrame
    np.ndarray
    np.ndarray
    """

    df = test_df.copy()

    df = add_minimal_clean_columns(df)

    emb_q1, emb_q2 = encode_questions(
        df,
        embedding_model
    )

    df = add_embedding_distance_features(
        df,
        emb_q1,
        emb_q2
    )

    return df, emb_q1, emb_q2

def build_embedding_matrix(
    df,
    emb_q1,
    emb_q2,
    feature_columns,
):
    """
    Builds feature matrix for LR/XGBoost.
    """

    handcrafted = df[feature_columns].to_numpy()

    X = np.hstack([
        emb_q1,
        emb_q2,
        handcrafted
    ])

    return X

def prepare_test_dataset(df, tokenizer, max_length):
    """
    Prepare test dataset for RoBERTa evaluation or inference.

    If the dataframe contains 'is_duplicate', it is renamed to 'labels'
    so that Trainer can automatically compute evaluation metrics.
    """

    columns = ["question1_clean", "question2_clean"]

    if "is_duplicate" in df.columns:
        columns.append("is_duplicate")

    test_dataset = Dataset.from_pandas(
        df[columns],
        preserve_index=False
    )

    def tokenize(batch):
        return tokenizer(
            batch["question1_clean"],
            batch["question2_clean"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    test_dataset = test_dataset.map(
        tokenize,
        batched=True
    )

    remove_cols = ["question1_clean", "question2_clean"]

    if "is_duplicate" in test_dataset.column_names:
        test_dataset = test_dataset.rename_column(
            "is_duplicate",
            "labels"
        )

    test_dataset = test_dataset.remove_columns(remove_cols)

    test_dataset.set_format(type="torch")

    return test_dataset


def predict_model(inputs, model):
    """
    Makes predictions using a trained classification model.

    Parameters
    ----------
    inputs : array-like
        Feature matrix.

    model : sklearn estimator
        Trained classification model.

    Returns
    -------
    pd.DataFrame
        Predictions and probabilities.
    """

    start_time = time.perf_counter()

    predictions = model.predict(inputs)
    probabilities = model.predict_proba(inputs)[:, 1]

    inference_time = time.perf_counter() - start_time

    labels = np.where(predictions == 1, "Duplicate", "Not duplicate")

    results = pd.DataFrame({
        "prediction": predictions,
        "probability": probabilities,
        "labels": labels
    })

    print(f"Inference time: {inference_time:.4f} s")

    return results


def predict_roberta(trainer, dataset):
    """
    Makes predictions using a fine-tuned BERT/RoBERTa model.

    Parameters
    ----------
    trainer : transformers.Trainer

    dataset : HuggingFace Dataset

    Returns
    -------
    pd.DataFrame
    """

    start_time = time.perf_counter()

    outputs = trainer.predict(dataset)

    logits = outputs.predictions

    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=1
    ).numpy()

    predictions = np.argmax(probabilities, axis=1)

    inference_time = time.perf_counter() - start_time

    labels = np.where(
        predictions == 1,
        "Duplicate",
        "Not duplicate"
    )

    results = pd.DataFrame({
        "prediction": predictions,
        "probability": probabilities[:, 1],
        "label": labels
    })

    print(f"Inference time: {inference_time:.4f} s")

    return results