import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import torch

from scipy.special import softmax
from sklearn.metrics import log_loss, f1_score, classification_report, confusion_matrix, roc_auc_score


def train_model(model, X_train, y_train):
    """
    Trains a model and measures training time.

    Returns
    -------
    model
        Trained model.

    training_time : float
        Training time in seconds.
    """

    start_time = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_time

    print(f"Training time: {training_time:.2f} seconds")

    return model, training_time


def compute_metrics(eval_pred):
    """
    Calculates evaluation metrics for binary classification.

    Returns
    -------
    dict
        Log Loss and F1-score.
    """

    logits, labels = eval_pred

    probabilities = softmax(logits, axis=1)

    predictions = np.argmax(probabilities, axis=1)

    return {
        "log_loss": log_loss(labels, probabilities),
        "f1": f1_score(labels, predictions),
    }




def classify_analysis(targets, inputs, model, name="Validation"):
    """
    Evaluates a binary classification model.

    Parameters
    ----------
    targets : array-like
        True labels.

    inputs : array-like
        Feature matrix.

    model : sklearn estimator
        Trained classification model.

    name : str, default="Validation"
        Dataset name used in plots and printed output.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    # Measure inference time
    start_time = time.perf_counter()

    predictions = model.predict(inputs)
    probabilities = model.predict_proba(inputs)[:, 1]

    inference_time = time.perf_counter() - start_time

    # Metrics
    ll = log_loss(targets, probabilities)
    auc = roc_auc_score(targets, probabilities)
    f1 = f1_score(targets, predictions)

    print(name)
    print(f"Log Loss      : {ll:.4f}")
    print(f"ROC-AUC       : {auc:.4f}")
    print(f"F1-score      : {f1:.4f}")
    print(f"Inference time: {inference_time:.4f} s")

    # Confusion Matrix
    cm = confusion_matrix(targets, predictions)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not duplicate", "Duplicate"],
        yticklabels=["Not duplicate", "Duplicate"],
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    # Classification Report
    print("\nClassification Report:\n")
    print(classification_report(targets, predictions))

    return {
        "log_loss": ll,
        "roc_auc": auc,
        "f1_score": f1,
        "inference_time": inference_time,
    }




def classify_analysis_bert(trainer, dataset, name="Validation"):
    """
    Evaluates a fine-tuned BERT model.
    """

    predictions = trainer.predict(dataset)

    logits = predictions.predictions

    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=1
    ).numpy()

    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.array(dataset["labels"])

    ll = log_loss(y_true, probabilities)
    auc = roc_auc_score(y_true, probabilities[:, 1])
    f1 = f1_score(y_true, y_pred)

    inference_time = predictions.metrics["test_runtime"]

    print(name)

    print(f"Log Loss      : {ll:.4f}")
    print(f"ROC-AUC       : {auc:.4f}")
    print(f"F1-score      : {f1:.4f}")
    print(f"Inference time: {inference_time:.4f} s")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not duplicate", "Duplicate"],
        yticklabels=["Not duplicate", "Duplicate"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{name} Confusion Matrix")
    plt.show()

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, digits=4))

    return {
        "log_loss": ll,
        "roc_auc": auc,
        "f1_score": f1,
        "inference_time": inference_time,
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probabilities,
    }




def error_analysis(df, X, model):
    """
    Performs error analysis for a binary classifier.

    Parameters
    ----------
    df : pandas.DataFrame
        Validation dataframe.

    X : array-like
        Feature matrix.

    model : sklearn estimator
        Trained model.

    Returns
    -------
    dict
        Dictionary containing TP, TN, FP and FN dataframes.
    """

    results = df.copy()

    results["prediction"] = model.predict(X)
    results["probability"] = model.predict_proba(X)[:, 1]

    tp = results[
        (results["is_duplicate"] == 1) &
        (results["prediction"] == 1)
    ]

    tn = results[
        (results["is_duplicate"] == 0) &
        (results["prediction"] == 0)
    ]

    fp = results[
        (results["is_duplicate"] == 0) &
        (results["prediction"] == 1)
    ]

    fn = results[
        (results["is_duplicate"] == 1) &
        (results["prediction"] == 0)
    ]

    print("Error Analysis")

    print(f"TP : {len(tp)}")
    print(f"TN : {len(tn)}")
    print(f"FP : {len(fp)}")
    print(f"FN : {len(fn)}")

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def error_analysis_bert(df, analysis_results):
    """
    Performs error analysis for a fine-tuned BERT model.
    """

    results = df.copy()

    results["prediction"] = analysis_results["y_pred"]
    results["probability"] = analysis_results["probabilities"][:, 1]

    tp = results[
        (results["is_duplicate"] == 1) &
        (results["prediction"] == 1)
    ]

    tn = results[
        (results["is_duplicate"] == 0) &
        (results["prediction"] == 0)
    ]

    fp = results[
        (results["is_duplicate"] == 0) &
        (results["prediction"] == 1)
    ]

    fn = results[
        (results["is_duplicate"] == 1) &
        (results["prediction"] == 0)
    ]

    print("Error Analysis\n")

    print(f"TP : {len(tp)}")
    print(f"TN : {len(tn)}")
    print(f"FP : {len(fp)}")
    print(f"FN : {len(fn)}")

    return {
        "results": results,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }