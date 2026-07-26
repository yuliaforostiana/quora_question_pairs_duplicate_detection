from collections import Counter
from typing import Iterable, Optional

import contractions
import numpy as np
import pandas as pd
import re
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats



# 1. Text Preprocessing

def preprocess_text(
    text: str,
    remove_numbers: bool = False,
    remove_stopwords: bool = False,
    expand_contractions_flag: bool = True,
    stop_words: Optional[set] = None,
) -> list:
    """
    Clean and tokenize a single text string.

    Steps applied (in order): lowercasing, contraction expansion,
    optional digit removal, punctuation removal, whitespace normalization,
    tokenization, optional stopword removal.

    Parameters
    ----------
    text : str
        Input text to preprocess. Non-string input returns an empty list.
    remove_numbers : bool, default=False
        If True, strip digits from the text before tokenizing.
    remove_stopwords : bool, default=False
        If True, remove English stopwords from the resulting tokens.
        Requires `stop_words` to be provided.
    expand_contractions_flag : bool, default=True
        If True, expand contractions (e.g. "don't" -> "do not",
        "ain't" -> "am not") before punctuation is stripped. This avoids
        splitting a contraction into a meaningless token pair
        (e.g. "don" + "t"). Uses the `contractions` library, which covers
        both standard and informal contractions -- relevant for
        user-generated text such as Quora questions.
    stop_words : set, optional
        Set of stopwords to remove when `remove_stopwords=True`.

    Returns
    -------
    list of str
        Preprocessed tokens.
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

    if remove_stopwords:
        if stop_words is None:
            raise ValueError("stop_words must be provided when remove_stopwords=True")
        tokens = [word for word in tokens if word not in stop_words]

    return tokens



# 2. Corpus-level Statistics


def corpus_word_frequency(
    df: pd.DataFrame,
    token_columns: Iterable[str] = ("question1_tokens", "question2_tokens"),
) -> Counter:
    """
    Count word frequencies across the entire corpus.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing preprocessed token columns.
    token_columns : iterable of str, default=("question1_tokens", "question2_tokens")
        Columns containing tokenized text.

    Returns
    -------
    collections.Counter
        Word -> frequency mapping across all specified columns.
    """

    all_tokens = []

    for col in token_columns:
        for tokens in df[col]:
            all_tokens.extend(tokens)

    return Counter(all_tokens)


def frequency_df(counter: Counter) -> pd.DataFrame:
    """
    Convert a word-frequency Counter into a sorted DataFrame.

    Parameters
    ----------
    counter : collections.Counter
        Word frequencies, e.g. from `corpus_word_frequency`.

    Returns
    -------
    pandas.DataFrame
        Columns: word, frequency, share_percent (frequency's share of the
        total token count), sorted by frequency descending.
    """

    total_words = sum(counter.values())

    freq_df = (
        pd.DataFrame(counter.items(), columns=["word", "frequency"])
        .sort_values("frequency", ascending=False)
        .reset_index(drop=True)
    )

    freq_df["share_percent"] = ((freq_df["frequency"] / total_words) * 100).round(4)

    return freq_df


def rare_words_ratio(counter: Counter, threshold: int = 1) -> float:
    """
    Compute the share of the vocabulary made up of rare words.

    A word is "rare" if it occurs `threshold` times or fewer in the corpus
    (hapax legomena when threshold=1). This measures vocabulary sparsity,
    not the share of "unique" words in the everyday sense.

    Parameters
    ----------
    counter : collections.Counter
        Word frequencies.
    threshold : int, default=1
        Maximum frequency for a word to be considered rare.

    Returns
    -------
    float
        Rare words / total vocabulary size, in [0, 1].
    """

    rare = sum(freq <= threshold for freq in counter.values())
    total = len(counter)

    return rare / total



# 3. Question-level statistics


def question_stats(
    df: pd.DataFrame,
    token_columns: Iterable[str] = ("question1_tokens", "question2_tokens"),
) -> pd.DataFrame:
    """
    Compute per-question descriptive statistics.

    For each token column, derives the matching original text column name
    by stripping the "_tokens" suffix (e.g. "question1_tokens" to
    "question1") and computes character count, word count, unique word
    count, unique word ratio, average word length, and max word length.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing both the original text columns and their
        corresponding tokenized columns.
    token_columns : iterable of str, default=("question1_tokens", "question2_tokens")
        Columns containing tokenized text.

    Returns
    -------
    pandas.DataFrame
        Copy of `df` with the added per-question statistic columns.
    """

    result = df.copy()

    for token_col in token_columns:
        question_col = token_col.replace("_tokens", "")
        tokens = result[token_col]

        result[f"{question_col}_char_count"] = (
            result[question_col].fillna("").str.len()
        )

        result[f"{question_col}_word_count"] = tokens.apply(len)

        result[f"{question_col}_unique_word_count"] = tokens.apply(
            lambda words: len(set(words))
        )

        result[f"{question_col}_unique_ratio"] = (
            result[f"{question_col}_unique_word_count"]
            / result[f"{question_col}_word_count"].replace(0, pd.NA)
        )

        result[f"{question_col}_avg_word_length"] = tokens.apply(
            lambda words: sum(len(w) for w in words) / len(words) if words else 0
        )

        result[f"{question_col}_max_word_length"] = tokens.apply(
            lambda words: max(len(w) for w in words) if words else 0
        )

    return result



# 4. Pairwise Overlap Features


def word_overlap_features(
    df: pd.DataFrame,
    token_col1: str = "question1_tokens",
    token_col2: str = "question2_tokens",
) -> pd.DataFrame:
    """
    Compute word-overlap features between a pair of questions.

    Features
    --------
    common_word_count : int
        Number of shared unique words.
    jaccard : float, [0, 1]
        Penalizes words present in only one question
        (indirectly also penalizes a length difference between the two).
    common_ratio : float, [0, 1]
        Softer than jaccard: does not penalize a
        question being longer simply because it contains extra context.
    len_diff_words : int
        Absolute difference in word count between the two questions.
    len_diff_chars : int
        Absolute difference in character count between the two questions.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the token columns and the original
        question1/question2 text columns.
    token_col1, token_col2 : str
        Names of the tokenized question columns.

    Returns
    -------
    pandas.DataFrame
        Copy of `df` with the overlap feature columns added.
    """

    out = df.copy()

    common_word_counts = []
    jaccard_scores = []
    common_ratios = []

    for tokens1, tokens2 in zip(out[token_col1], out[token_col2]):
        words1, words2 = set(tokens1), set(tokens2)

        common_count = len(words1 & words2)
        total_unique_count = len(words1 | words2)
        smaller_question_size = min(len(words1), len(words2))

        common_word_counts.append(common_count)
        jaccard_scores.append(common_count / total_unique_count if total_unique_count else 0.0)
        common_ratios.append(common_count / smaller_question_size if smaller_question_size else np.nan)

    out["common_word_count"] = common_word_counts
    out["jaccard"] = jaccard_scores
    out["common_ratio"] = common_ratios

    out["len_diff_words"] = (out[token_col1].apply(len) - out[token_col2].apply(len)).abs()
    out["len_diff_chars"] = (out["question1"].fillna("").str.len() - out["question2"].fillna("").str.len()).abs()

    return out



# 5. Visualization


def distribution_plot(column1: pd.Series, column2: Optional[pd.Series] = None, bins: int = 30) -> None:
    """
    Plot the distribution of one or two numerical variables.

    Single-variable mode shows a histogram, a QQ plot against the normal
    distribution, and a boxplot. Two-variable mode shows an overlaid
    histogram and a side-by-side boxplot for comparison (e.g. duplicate
    vs. non-duplicate class).

    Parameters
    ----------
    column1 : pandas.Series
        First variable to plot. NaNs are dropped before plotting.
    column2 : pandas.Series, optional
        Second variable to plot alongside `column1` for comparison.
    bins : int, default=30
        Number of histogram bins.
    """

    column1 = column1.dropna()

    if column2 is not None:
        column2 = column2.dropna()

    # single variable
    if column2 is None:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'Distribution: {column1.name}')

        sns.histplot(column1, bins=bins, ax=axs[0])
        axs[0].set_title('Histogram')

        stats.probplot(column1, dist='norm', plot=axs[1])
        axs[1].set_title('QQ Plot')

        sns.boxplot(x=column1, ax=axs[2])
        axs[2].set_title('Boxplot')

    # two variables
    else:
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{column1.name} vs {column2.name}')

        sns.histplot(column1, bins=bins, stat='density', alpha=0.5, label=column1.name, ax=axs[0])
        sns.histplot(column2, bins=bins, stat='density', alpha=0.5, label=column2.name, ax=axs[0])
        axs[0].legend()
        axs[0].set_title('Histogram')

        plot_df = pd.concat(
            [
                pd.DataFrame({'Variable': column1.name, 'Value': column1}),
                pd.DataFrame({'Variable': column2.name, 'Value': column2}),
            ],
            ignore_index=True,
        )

        sns.boxplot(data=plot_df, x='Variable', y='Value', ax=axs[1])
        axs[1].set_title('Boxplot')

    plt.tight_layout()
    plt.show()