import json
import re
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx
import contractions

import nltk
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity, paired_cosine_distances, paired_euclidean_distances
from scipy.sparse import vstack

nltk.download('stopwords')


# Data cleaning


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



# Splitting data into train and validation


def build_question_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Step 1: builds the question graph.

    Each unique question (qid) is a graph node.
    Each pair (qid1, qid2) in the dataset is an edge between the two nodes.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the qid1, qid2 columns.

    Returns
    -------
    networkx.Graph
    """
    graph = nx.Graph()
    graph.add_edges_from(zip(df["qid1"], df["qid2"]))

    print(f"Number of nodes (unique questions): {graph.number_of_nodes()}")
    print(f"Number of edges (pairs of questions): {graph.number_of_edges()}")

    return graph


def get_connected_components(graph: nx.Graph) -> list:
    """
    Step 2: finds the connected components of the graph.

    A component is a group of questions connected to each other via a chain
    of shared pairs (even if not all of them are directly in the same pair).

    Parameters
    ----------
    graph : networkx.Graph

    Returns
    -------
    list of set
        Each set is one component (the set of qids that belong to it).
    """
    components = list(nx.connected_components(graph))
    sizes = pd.Series([len(c) for c in components])

    print(f"Number of components: {len(components)}")
    print(f"Component size -- min: {sizes.min()}, "
          f"median: {sizes.median()}, max: {sizes.max()}")
    print("Component size distribution (percentiles):")
    print(sizes.describe(percentiles=[0.5, 0.9, 0.99, 0.999]).round(2))

    # how many components consist of just a single pair (the simplest, most common case)
    single_pair_components = (sizes == 2).sum()
    print(f"Number of components with exactly 2 questions (one isolated pair): "
          f"{single_pair_components} ({single_pair_components / len(components):.1%})")

    return components


def build_qid_to_component_map(components: list) -> dict:
    """
    Step 3: builds a qid -> component_id dictionary for fast lookup.

    Parameters
    ----------
    components : list of set
        Result of get_connected_components.

    Returns
    -------
    dict
        {qid: component_id}
    """
    qid_to_component = {
        qid: component_id
        for component_id, qids in enumerate(components)
        for qid in qids
    }

    print(f"Mapped {len(qid_to_component)} qid "
          f" to {len(components)} component")

    return qid_to_component


def split_components_into_train_val(
    n_components: int,
    *,
    random_state: int,
    val_size: float = 0.15,
) -> tuple:
    """
    Step 4: splits component INDICES (0, 1, 2, ..., n_components-1) into train/val.

    Important: it is the list of components that gets split, not the
    dataframe rows -- so the actual share of ROWS in val may differ from
    val_size if components have different sizes (see the explanation in
    assign_split_labels).

    Parameters
    ----------
    n_components : int
        Total number of components.
    random_state : int
        Must be passed explicitly by the caller (e.g. defined once in the
        notebook and reused across all calls in the pipeline).
    val_size : float, default=0.15
        Share of COMPONENTS (not rows!) that goes to val.

    Returns
    -------
    tuple of set
        (train_component_ids, val_component_ids)
    """
    all_component_ids = np.arange(n_components)

    train_component_ids, val_component_ids = train_test_split(
        all_component_ids, test_size=val_size, random_state=random_state
    )

    print(f"Number of components in train: {len(train_component_ids)}")
    print(f"Number of components in val: {len(val_component_ids)}")

    return set(train_component_ids), set(val_component_ids)


def assign_split_labels(
    df: pd.DataFrame,
    qid_to_component: dict,
    val_component_ids: set,
) -> pd.DataFrame:
    """
    Step 5: adds a 'split' column (train/val) to the dataframe based on
    which component question qid1 of each row belongs to.

    Parameters
    ----------
    df : pandas.DataFrame
    qid_to_component : dict
        Result of build_qid_to_component_map.
    val_component_ids : set
        Result of split_components_into_train_val.

    Returns
    -------
    pandas.DataFrame
        A copy of df with the added 'split' column.
    """
    df = df.copy()

    component_id_per_row = df["qid1"].map(qid_to_component)
    df["split"] = np.where(
        component_id_per_row.isin(val_component_ids), "val", "train"
    )

    # -------- Diagnostics: how closely the actual split matches the expected one --------
    row_counts = df["split"].value_counts()
    actual_val_share = row_counts.get("val", 0) / len(df)

    print("Number of rows in each split:")
    print(row_counts)
    print(f"Actual share val (by rows): {actual_val_share:.1%}")

    print("\nClass balance (is_duplicate proportion) in each split:")
    print(df.groupby("split")["is_duplicate"].mean().round(4).rename("duplicate_share"))

    return df



# Preprocess text


_lemmatizer = WordNetLemmatizer()


def _nltk_pos_to_wordnet_pos(nltk_tag: str) -> str:
    """Converts an nltk.pos_tag part-of-speech tag into WordNetLemmatizer's format."""
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize_tokens(tokens: list) -> list:
    """
    Lemmatizes tokens ("learning", "learns" -> "learn") taking part of
    speech into account (POS tagging) -- without this, WordNetLemmatizer
    treats everything as a noun.

    KNOWN LIMITATION: does not handle suppletive forms (good/better/best) --
    these aren't simple morphology, but separate exception words.
    """
    if not tokens:
        return []
    tagged = nltk.pos_tag(tokens)
    return [_lemmatizer.lemmatize(w, _nltk_pos_to_wordnet_pos(t)) for w, t in tagged]


def preprocess_text(
    text: str,
    remove_numbers: bool = False,
    remove_stopwords: bool = False,
    expand_contractions_flag: bool = True,
    lemmatize: bool = False,
    stop_words: Optional[set] = None,
) -> list:
    """
    Cleans and tokenizes text.

    Steps (in order): lowercase, expand contractions, (optionally)
    remove numbers, remove punctuation, normalize whitespace,
    tokenize, (optionally) lemmatize, (optionally) remove stop words.

    Parameters
    ----------
    text : str
    remove_numbers : bool, default=False
    remove_stopwords : bool, default=False
        Requires stop_words.
    expand_contractions_flag : bool, default=True
        Only for the classic track (TF-IDF/BoW). Do NOT use for
        BERT/sentence-transformers -- they need natural text.
    lemmatize : bool, default=False
        Only for the classic track, for the same reasons.
    stop_words : set, optional

    Returns
    -------
    list of str
    """
    if not isinstance(text, str):
        return []

    text = text.lower()

    if expand_contractions_flag:
        text = contractions.fix(text)

    if remove_numbers:
        text = re.sub(r'\d+', ' ', text)

    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()

    if lemmatize:
        tokens = lemmatize_tokens(tokens)

    if remove_stopwords:
        if stop_words is None:
            raise ValueError("stop_words must be provided when remove_stopwords=True")
        tokens = [w for w in tokens if w not in stop_words]

    return tokens



# Feature engineering


def word_overlap_features(
    df: pd.DataFrame, token_col1: str = "question1_tokens", token_col2: str = "question2_tokens",
) -> pd.DataFrame:
    """
    Word overlap features between the pair of questions:
    common_word_count, jaccard, common_ratio, len_diff_words, len_diff_chars.
    For the detailed formulas, see the earlier explanations in the EDA.
    """
    out = df.copy()
    common_word_counts, jaccard_scores, common_ratios = [], [], []

    for tokens1, tokens2 in zip(out[token_col1], out[token_col2]):
        w1, w2 = set(tokens1), set(tokens2)
        common = len(w1 & w2)
        total = len(w1 | w2)
        smaller = min(len(w1), len(w2))

        common_word_counts.append(common)
        jaccard_scores.append(common / total if total else 0.0)
        common_ratios.append(common / smaller if smaller else np.nan)

    out["common_word_count"] = common_word_counts
    out["jaccard"] = jaccard_scores
    out["common_ratio"] = common_ratios
    out["len_diff_words"] = (out[token_col1].apply(len) - out[token_col2].apply(len)).abs()
    out["len_diff_chars"] = (out["question1"].fillna("").str.len() - out["question2"].fillna("").str.len()).abs()

    return out


def add_shared_features(
    df: pd.DataFrame,
    remove_stopwords: bool = True,
    stop_words: Optional[set] = None,
) -> pd.DataFrame:
    """
    Tokenizes (with lemmatization) and computes engineered overlap features.

    Parameters
    ----------
    df : pandas.DataFrame
    remove_stopwords : bool, default=True
        Whether to remove stop words during tokenization for the
        engineered features. Test both options (see the call example below).
    stop_words : set, optional
        Required only if remove_stopwords=True. Define this in the
        notebook (e.g. from nltk.corpus.stopwords) and pass it in.
    """
    df = df.copy()

    df["question1_tokens"] = df["question1"].apply(
        lambda t: preprocess_text(t, remove_numbers=False, remove_stopwords=remove_stopwords,
                                   lemmatize=True, stop_words=stop_words)
    )
    df["question2_tokens"] = df["question2"].apply(
        lambda t: preprocess_text(t, remove_numbers=False, remove_stopwords=remove_stopwords,
                                   lemmatize=True, stop_words=stop_words)
    )

    df = word_overlap_features(df)
    print(f"Processed {len(df)} rows, remove_stopwords={remove_stopwords}")
    return df



# Creating dataset 1: TF-IDF


def tokens_to_text(df: pd.DataFrame) -> pd.DataFrame:
    """Joins the tokens (from add_shared_features) back into a string for TfidfVectorizer."""
    df = df.copy()
    df["question1_clean"] = df["question1_tokens"].apply(lambda t: " ".join(t))
    df["question2_clean"] = df["question2_tokens"].apply(lambda t: " ".join(t))
    return df


def fit_tfidf_vectorizer(train_df: pd.DataFrame) -> TfidfVectorizer:
    """Fits TF-IDF ONLY on train (otherwise -- data leakage through the vocabulary/IDF)."""
    train_texts = pd.concat([train_df["question1_clean"], train_df["question2_clean"]])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)
    vectorizer.fit(train_texts)
    print(f"dictionary size: {len(vectorizer.vocabulary_)}")
    return vectorizer


def add_tfidf_cosine(df: pd.DataFrame, vectorizer: TfidfVectorizer):
    """Transform (fit only on train!) + cosine similarity between q1 and q2."""
    df = df.copy()
    q1_vec = vectorizer.transform(df["question1_clean"])
    q2_vec = vectorizer.transform(df["question2_clean"])

    df["tfidf_cosine"] = np.array([
        cosine_similarity(q1_vec[i], q2_vec[i])[0, 0] for i in range(q1_vec.shape[0])
    ])
    return df, q1_vec, q2_vec


def fit_svd(train_q1_vec, train_q2_vec, n_components: int, random_state: int) -> TruncatedSVD:
    """
    Compresses the sparse TF-IDF matrix into a small number of dense
    values.

    Parameters
    ----------
    n_components : int
        Target number of SVD components. Define in the notebook.
    random_state : int
        Define in the notebook and reuse across the pipeline.
    """
    n_features = train_q1_vec.shape[1]
    if n_components >= n_features:
        n_components = max(1, n_features - 1)
        print(f"n_components reduced to {n_components} (small dictionary)")

    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    svd.fit(vstack([train_q1_vec, train_q2_vec]))
    print(f"explained variance: {svd.explained_variance_ratio_.sum():.2%}")
    return svd


def add_svd_features(df: pd.DataFrame, q1_vec, q2_vec, svd: TruncatedSVD) -> pd.DataFrame:
    """Adds the q1/q2 SVD components as separate numeric columns."""
    df = df.reset_index(drop=True).copy()
    q1_svd, q2_svd = svd.transform(q1_vec), svd.transform(q2_vec)
    n = svd.n_components
    cols1 = pd.DataFrame(q1_svd, columns=[f"q1_svd_{i}" for i in range(n)])
    cols2 = pd.DataFrame(q2_svd, columns=[f"q2_svd_{i}" for i in range(n)])
    return pd.concat([df, cols1, cols2], axis=1)


def prepare_dataset1(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    use_svd: bool = True,
    svd_n_components: Optional[int] = None,
    random_state: Optional[int] = None,
):
    """
    Full pipeline for Dataset 1. train_df/val_df must have already gone
    through add_shared_features().

    use_svd=False -- sufficient if you're only planning to use logistic
    regression (linear models work well directly on sparse TF-IDF).

    Parameters
    ----------
    svd_n_components : int, optional
        Required if use_svd=True. Define in the notebook.
    random_state : int, optional
        Required if use_svd=True. Define in the notebook.

    Returns
    -------
    (train_df, val_df, vectorizer, svd_or_None)
        Save vectorizer and svd via joblib for the modeling/deployment file.
    """
    train_df = tokens_to_text(train_df)
    val_df = tokens_to_text(val_df)

    vectorizer = fit_tfidf_vectorizer(train_df)
    train_df, train_q1_vec, train_q2_vec = add_tfidf_cosine(train_df, vectorizer)
    val_df, val_q1_vec, val_q2_vec = add_tfidf_cosine(val_df, vectorizer)

    svd = None
    if use_svd:
        if svd_n_components is None or random_state is None:
            raise ValueError("svd_n_components and random_state must be provided when use_svd=True")
        svd = fit_svd(train_q1_vec, train_q2_vec, n_components=svd_n_components, random_state=random_state)
        train_df = add_svd_features(train_df, train_q1_vec, train_q2_vec, svd)
        val_df = add_svd_features(val_df, val_q1_vec, val_q2_vec, svd)

    return train_df, val_df, vectorizer, svd



# Creating dataset 2: sentence embeddings


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


def prepare_dataset2(train_df: pd.DataFrame, val_df: pd.DataFrame, model_name: str):
    """
    Full pipeline for Dataset 2. train_df/val_df must have already gone
    through add_shared_features().

    Parameters
    ----------
    model_name : str
        Sentence-transformers model name. Define in the notebook.

    Returns
    -------
    (train_df, val_df, model_name, embeddings_dict)
        Save model_name -- at inference time you must load the SAME model.
        embeddings_dict -- the raw vectors, in case you want to feed them
        directly into a model.
    """

    from sentence_transformers import SentenceTransformer

    train_df = add_minimal_clean_columns(train_df)
    val_df = add_minimal_clean_columns(val_df)

    model = SentenceTransformer(model_name)

    train_q1_emb, train_q2_emb = encode_questions(train_df, model)
    val_q1_emb, val_q2_emb = encode_questions(val_df, model)

    train_df = add_embedding_distance_features(train_df, train_q1_emb, train_q2_emb)
    val_df = add_embedding_distance_features(val_df, val_q1_emb, val_q2_emb)

    embeddings = {
        "train_q1": train_q1_emb, "train_q2": train_q2_emb,
        "val_q1": val_q1_emb, "val_q2": val_q2_emb,
    }
    return train_df, val_df, model_name, embeddings



# Preparing BERT


def determine_max_length(train_df: pd.DataFrame, model_checkpoint: str, percentile: float = 0.99) -> int:
    """
    max_length based on the actual token-length distribution of the pair,
    using the CHOSEN tokenizer. Computed ONLY on train (so as not to peek
    at val).
    """

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    token_lengths = train_df.apply(
        lambda r: len(tokenizer.encode(r["question1_clean"], r["question2_clean"], add_special_tokens=True)),
        axis=1,
    )

    raw_value = int(token_lengths.quantile(percentile))
    max_length = ((raw_value + 7) // 8) * 8

    print(f"{percentile:.0%} percentile: {raw_value} tokens")
    print(f"Recommended max_length: {max_length}")
    print(token_lengths.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(1))

    return max_length


def prepare_dataset3(train_df: pd.DataFrame, val_df: pd.DataFrame, model_checkpoint: str):
    """
    Full pipeline for Dataset 3 (text preparation only -- "on the fly"
    tokenization already happens in the modeling file).

    Parameters
    ----------
    model_checkpoint : str
        HuggingFace checkpoint name. Define in the notebook.

    Returns
    -------
    (train_df, val_df, config)
        config -- {"model_checkpoint": ..., "max_length": ...}, save it
        as JSON (save_config) for the modeling/deployment file.
    """
    train_df = add_minimal_clean_columns(train_df)
    val_df = add_minimal_clean_columns(val_df)

    max_length = determine_max_length(train_df, model_checkpoint)

    config = {"model_checkpoint": model_checkpoint, "max_length": max_length}
    return train_df, val_df, config


def save_config(config: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved: {path}")