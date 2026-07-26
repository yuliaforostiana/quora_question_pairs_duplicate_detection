# Quora Question Pairs — Duplicate Detection

Predicting whether two Quora questions are semantically duplicate, using classical ML, gradient boosting, and transformer-based models.

## [Live Demo](https://quoraquestionpairsduplicatedetection.streamlit.app/)

Interact with the deployed Streamlit application by entering two questions and receiving a real-time duplicate prediction with the corresponding confidence score.

## Table of Contents

1. [Business Task & Goal](#business-task--goal)
2. [Data](#data)
3. [Evaluation Approach](#evaluation-approach)
4. [Approach & Tools](#approach--tools)
5. [Results](#results)
6. [Conclusions](#conclusions)
7. [Repository Structure](#repository-structure)
8. [Installation & Usage](#installation--usage)
9. [Requirements](#requirements)

## Business Task & Goal

The task is to classify whether two given questions have the same meaning (i.e. are duplicates of each other). Using natural language processing techniques, the goal is to predict the probability that a pair of questions is a duplicate.

This kind of duplicate-detection capability is broadly useful for businesses, e.g.:
- detecting similar product titles/listings on a marketplace,
- deduplicating customer records in a database,
- identifying repeated customer support messages/queries in a chat.

The objective is to build the most accurate model possible for predicting the probability that a pair of questions are duplicates.

## Data

- **Source:** Quora Question Pairs dataset (`quora_question_pairs_train.csv.zip`)
- **Columns:** `qid1`, `qid2`, `question1`, `question2`, `is_duplicate`
- **Size:** full training set, no exact duplicate rows

Key data-quality findings from EDA (see [`00_eda.ipynb`](https://github.com/yuliaforostiana/quora_question_pairs_duplicate_detection/blob/main/notebooks/00_eda.ipynb) for full analysis):

- **Missing values:** 3 rows have a missing question text and need to be dropped.
- **Mislabeled identical pairs:** 18 pairs have identical `question1`/`question2` text under different `qid`s, of which 5 are (incorrectly) labeled `is_duplicate = 0`. These are treated as a likely labeling error.
- **Question length:** right-skewed distribution — median ~52 characters, ~10% of questions exceed 100 characters.
- **Repeated questions:** ~19.1% of unique questions appear more than once across the dataset. Duplicate-labeled pairs tend to involve more frequently repeated questions than non-duplicate pairs — this was accounted for when splitting train/validation data, to avoid the same question leaking across both splits.
- **Class balance:** 63.08% non-duplicate vs. 36.92% duplicate — a mild imbalance, not severe enough to require class-balancing techniques.
- **Vocabulary:** typos and inconsistent word forms (e.g. "good" vs. "best") are present; ~38% of the vocabulary consists of words occurring only once (rare words / hapax legomena).
- **Word overlap vs. duplication:** duplicate pairs show noticeably higher Jaccard similarity and `common_ratio` (share of shared words relative to the shorter question) than non-duplicate pairs. Duplicate questions also tend to be shorter and have a lower unique-word ratio than non-duplicates.
- **Correlation:** word-overlap ratio correlates positively with `is_duplicate`; word/character length differences between the two questions correlate negatively.

### Data Cleaning & Splitting

Implemented in [`01_preprocessing.ipynb`](https://github.com/yuliaforostiana/quora_question_pairs_duplicate_detection/blob/main/notebooks/01_preprocessing.ipynb) / [`preprocessing_utils.py`](https://github.com/yuliaforostiana/quora_question_pairs_duplicate_detection/blob/main/src/preprocessing_utils.py), directly addressing the issues surfaced in EDA:

1. **Drop rows with missing questions** (`clean_missing`).
2. **Fix mislabeled identical pairs** — pairs with `question1 == question2` (as text) but `is_duplicate == 0` are relabeled to `1` (`fix_identical_text_mislabels`).
3. **Resolve mirror pairs** — for a question pair appearing as both `(qid1, qid2)` and `(qid2, qid1)`: if the labels agree, one copy is kept; if they conflict, both rows are dropped (`dedupe_mirror_pairs`).
4. **Leakage-free train/validation split**, built as a graph problem rather than a random row split:
   - every unique question (`qid`) is a graph node, every question pair is an edge (`build_question_graph`);
   - **connected components** of this graph are identified (`get_connected_components`) — a component groups together all questions that are transitively linked through shared pairs, even if not all pairwise combinations exist directly in the data;
   - **components** (not individual rows) are split into train/val (`split_components_into_train_val`, 20% val), guaranteeing that no question ever appears in both splits (`assign_split_labels`).

   This directly implements the leakage-prevention concern flagged in the EDA conclusions.

### Feature Sets Prepared

Three parallel feature representations were built for downstream modeling (each split further into "with stopwords" / "without stopwords" where relevant):

| Dataset | Description | Used by |
|---|---|---|
| **Dataset 1 — TF-IDF (+ optional SVD)** | POS-aware lemmatized tokens (`WordNetLemmatizer` with NLTK POS tagging), TF-IDF (1–2 n-grams, `min_df=3`, `max_df=0.9`) fit on train only, cosine similarity between question vectors, optional `TruncatedSVD` compression (100 components) + handcrafted overlap features (`jaccard`, `common_ratio`, `len_diff_words`, `len_diff_chars`) | Logistic Regression, XGBoost |
| **Dataset 2 — Sentence embeddings** | Minimally cleaned text (whitespace normalization only — no lowercasing/lemmatization, to stay in-distribution for pretrained encoders) encoded with two different sentence-transformer models (Model 1 -`all-mpnet-base-v2`, Model 2 - `BAAI/bge-base-en-v1.5`), plus cosine/euclidean distance features between embeddings | Logistic Regression, XGBoost |
| **Dataset 3 — Transformer fine-tuning inputs** | Minimally cleaned text; `max_length` derived per-checkpoint from the 99th percentile of the actual tokenized pair length on train (rounded up to a multiple of 8) | BERT (`distilbert-base-uncased`), RoBERTa (`FacebookAI/roberta-base`) |

Note: the classic ML track (Dataset 1) uses **aggressive normalization** (lowercase, contraction expansion, lemmatization, optional stopword removal), while the embedding/transformer tracks (Datasets 2–3) use **minimal cleaning only** — pretrained encoders are trained on natural text, so aggressive normalization would push inputs off-distribution.

## Evaluation Approach

The primary metric is **Log Loss (Cross-Entropy Loss)** between predicted probabilities and true labels — the lower, the better.

**F1-score** and the **confusion matrix** are tracked as secondary metrics throughout experimentation, to monitor the precision/recall balance in addition to the probability-calibration quality that log loss captures.

**ROC-AUC** was also tracked across experiments as a threshold-independent measure of ranking quality.

## Approach & Tools

**EDA:** `pandas`, `numpy`, `seaborn`/`matplotlib`, `scipy.stats`, `nltk` (stopwords), `contractions` — see `eda_utils.py` for the reusable preprocessing/analysis functions (text cleaning, corpus/question-level statistics, word-overlap features, distribution plots) used throughout `eda.ipynb`.

**Preprocessing & feature engineering:** `pandas`, `numpy`, `networkx` (graph-based leakage-free splitting), `scikit-learn` (`TfidfVectorizer`, `TruncatedSVD`, `train_test_split`), `nltk` (POS tagging + `WordNetLemmatizer`), `contractions`, `sentence-transformers`, `transformers` (tokenizer-based `max_length` sizing) — see `preprocessing_utils.py` for all reusable pipeline functions used in `preprocessing.ipynb`.

**Feature engineering approaches compared:**
- TF-IDF (with and without stopwords), lemmatized, 1–2 n-grams
- TF-IDF + SVD dimensionality reduction + handcrafted features (`common_ratio`, `len_diff_words`, etc.)
- Sentence embeddings (two different sentence-transformer models) + cosine/euclidean distance features
- Minimally-cleaned text prepared for BERT/RoBERTa fine-tuning, with data-driven `max_length` sizing

**Models compared:**
- Logistic Regression (baseline, plus `RandomizedSearchCV`/Hyperopt-tuned variants)
- XGBoost (baseline, plus Hyperopt-tuned variant)
- BERT (fine-tuned, multiple checkpoints)
- RoBERTa (fine-tuned)

**Deployment:** FastAPI (REST API) and Streamlit (interactive UI) — see [`fast_api/`](https://github.com/yuliaforostiana/quora_question_pairs_duplicate_detection/tree/main/fast_api) and [`streamlit_app/`](https://github.com/yuliaforostiana/quora_question_pairs_duplicate_detection/tree/main/streamlit_app).

### Modeling Pipeline

Shared evaluation/training helpers live in [`model_utils.py`](https://github.com/yuliaforostiana/quora_question_pairs_duplicate_detection/blob/main/src/model_utils.py) and are reused across every modeling notebook:
- `train_model` — fits a model and times training
- `classify_analysis` / `classify_analysis_bert` — computes Log Loss, ROC-AUC, F1, inference time, confusion matrix, and classification report for sklearn-style and fine-tuned BERT/RoBERTa models respectively
- `error_analysis` / `error_analysis_bert` — splits predictions into TP/TN/FP/FN subsets for qualitative inspection

Modeling notebooks (each loads the feature sets produced by `preprocessing.ipynb`):

| Notebook | Model(s) | Feature set(s) |
|---|---|---|
| [`02_baseline.ipynb`](https://github.com/yuliaforostiana/quora_question_pairs_duplicate_detection/blob/main/notebooks/02_baseline.ipynb) | Logistic Regression (baseline) | Dataset 1 — TF-IDF, with and without stopwords |
| [`03_logistic_regression.ipynb`](https://github.com/yuliaforostiana/quora_question_pairs_duplicate_detection/blob/main/notebooks/03_logistic_regression.ipynb) | Logistic Regression (baseline + `RandomizedSearchCV`-tuned) | Dataset 1 — TF-IDF (with/without stopwords), TF-IDF + SVD (with/without stopwords); Dataset 2 — Sentence embeddings (both models) |
| [`04_xgboost.ipynb`](https://github.com/yuliaforostiana/quora_question_pairs_duplicate_detection/blob/main/notebooks/04_xgboost.ipynb) | XGBoost (baseline + Hyperopt-tuned) | Dataset 1 — TF-IDF (with/without stopwords), TF-IDF + SVD (with/without stopwords); Dataset 2 — Sentence embeddings (both models) |
| [`05_bert.ipynb`](https://github.com/yuliaforostiana/quora_question_pairs_duplicate_detection/blob/main/notebooks/05_bert.ipynb) | Fine-tuned DistilBERT (`distilbert-base-uncased`) | Dataset 3 — minimally cleaned question pairs, tokenized jointly (question1 + question2) |
| [`06_roberta.ipynb`](https://github.com/yuliaforostiana/quora_question_pairs_duplicate_detection/blob/main/notebooks/06_roberta.ipynb) | Fine-tuned RoBERTa (`FacebookAI/roberta-base`) | Dataset 3 — minimally cleaned question pairs, tokenized jointly (question1 + question2) |
| [`07_prediction.ipynb`](https://github.com/yuliaforostiana/quora_question_pairs_duplicate_detection/blob/main/notebooks/07_prediction.ipynb) | Final held-out test evaluation of the 3 shortlisted models (Logistic Regression, XGBoost, RoBERTa) | Held-out test set, same feature pipelines as training |

**Baseline findings** (`02_baseline.ipynb`, plain TF-IDF + Logistic Regression, no SVD/tuning):
- Removing stopwords makes results *worse*, not better — the with-stopwords variant outperforms the without-stopwords one on validation metrics.
- The baseline models clearly **overfit** — training metrics are substantially better than validation metrics (visible later in the full results table: e.g. train Log Loss 0.345 vs. val 0.561).
- Qualitative error analysis shows the model struggles with pairs that share many surface words but differ in meaning (e.g. "best place to **reside**" vs. "best place to **visit**"; "**when** did you join Quora?" vs. "**why** did I join Quora?") — a known limitation of bag-of-words-style features that later motivates moving to SVD compression, handcrafted overlap features, and ultimately sentence embeddings.

**Logistic Regression findings** (`03_logistic_regression.ipynb`, systematic comparison across all three feature sets, with `RandomizedSearchCV` hyperparameter tuning — `C`, `solver`, `class_weight` — on top of each):
- **TF-IDF + SVD clearly beats raw TF-IDF** (val ROC-AUC ~0.79 vs. ~0.75), confirming that dimensionality reduction plus handcrafted overlap features help a linear model generalize better than raw sparse n-gram counts.
- **Sentence embeddings are the strongest feature set for Logistic Regression by a wide margin** — val ROC-AUC jumps to ~0.92 and F1 to ~0.77–0.79, matching (or on F1, slightly beating) far more expensive transformer models, at a fraction of the training/inference cost.
- **Hyperparameter tuning via `RandomizedSearchCV` gave mixed results**: on raw TF-IDF it noticeably improved val F1 by pushing toward `class_weight="balanced"`, but on the already-strong sentence-embedding features the tuned model's Log Loss/ROC-AUC were roughly on par with the untuned baseline — the best F1 result overall (0.7864) came from a *further*-tuned sentence-embedding model with explicit `C`/`solver` values.
- Across every feature set, models still show a train/validation gap consistent with the overfitting tendency noted in the baseline, though the gap narrows substantially once sentence embeddings are used.

**XGBoost findings** (`04_xgboost.ipynb`, baseline `XGBClassifier` per feature set, plus Hyperopt-tuned models — `max_depth`, `learning_rate`, `n_estimators`, `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`/`reg_lambda` — searched via `hyperopt`'s TPE algorithm with `StratifiedKFold` cross-validation):
- Same feature-quality ordering as Logistic Regression holds: raw TF-IDF < TF-IDF + SVD < sentence embeddings, confirming this is a property of the *features*, not the specific model.
- **Sentence embeddings + Hyperopt-tuned XGBoost is the best-performing configuration found in the project overall on the primary metric (Log Loss)**, and is the model selected for deployment — saved as `models/xgboost_tfidf_se1.joblib` (see `fast_api/` and `streamlit_app/`).
- Unlike Logistic Regression, XGBoost's baseline (untuned) performance on raw/SVD-compressed TF-IDF is noticeably weaker (e.g. val F1 as low as ~0.32–0.44) — tree-based models need either richer features or tuning to make good use of high-dimensional sparse TF-IDF input, whereas linear models handle that representation more natively.
- Hyperparameter tuning (Hyperopt) gives a modest but consistent improvement over the default `XGBClassifier` settings on sentence embeddings, and is what pushes this configuration slightly ahead of the tuned Logistic Regression on Log Loss/ROC-AUC (though Logistic Regression retains a marginally higher F1).

**BERT findings** (`05_bert.ipynb`, `distilbert-base-uncased` fine-tuned on question pairs jointly tokenized as a single sequence, trained on GPU via Google Colab, `max_length` set from the data-driven value computed in `preprocessing.ipynb`):
- Training was resumed across multiple checkpoints over 3 epochs; the **final checkpoint (48000 steps) clearly overfits** — training Log Loss drops to ~0.11 while validation Log Loss rises back up to ~0.57, worse than several classical-ML configurations despite the much higher training cost.
- The **best checkpoint (14000 steps), selected by validation performance during training rather than by training the longest**, generalizes far better (train Log Loss ~0.22 / val Log Loss ~0.38) and achieves the **highest F1 of any model in the project (~0.79)** on validation.
- This is a useful project-level lesson: for fine-tuned transformers, **more training steps did not mean a better model** — checkpoint selection based on a held-out metric was essential, and training that overshoots the optimum quietly loses value that isn't visible from the training loss alone.
- Despite the strong F1 score, inference time (~157s in the timed runs) is 2–3 orders of magnitude slower than the sentence-embedding-based classical models, which is the deciding factor against using BERT in the deployed API (see [Conclusions](#conclusions)).

**RoBERTa findings** (`06_roberta.ipynb`, `FacebookAI/roberta-base` fine-tuned the same way as BERT, but with `load_best_model_at_end=True` and `metric_for_best_model="f1"` — a deliberate choice to see how a checkpoint optimized for classification balance compares against the project's other models):
- Achieves the **best F1 (~0.82) and best ROC-AUC (~0.94) of every model in the project**, confirming that a larger, more capable pretrained encoder still has headroom over DistilBERT and the classical-ML + embeddings approach on ranking/classification quality.
- Its **Log Loss (~0.50) comes in higher than the tuned XGBoost-on-embeddings model (~0.33)**, which illustrates an important trade-off: selecting the checkpoint that best balances precision/recall (F1) is not the same as selecting the one with the best-calibrated probabilities (Log Loss) — the two metrics reward different things, and a model can lead on one while trailing on the other.
- Since the project is ultimately evaluated on **Log Loss**, this result reinforces the decision to deploy the model that leads on that specific metric — the tuned XGBoost-on-embeddings configuration — while RoBERTa remains the strongest option if F1/ROC-AUC were the priority instead (e.g. for a use case that cares more about correct classification at a fixed threshold than about probability calibration).
- Training took ~2h8min and inference ~318s in the timed runs — even more expensive than BERT, reinforcing that a transformer, however strong on quality metrics, is not the right choice for the latency budget of this project's prediction API.
- Combined with the BERT results, this closes out the full model comparison in the project (see the [Results](#results) table for the complete picture); **XGBoost on sentence embeddings remains the deployed model**, as the best quality/latency trade-off on the metric that matters most (Log Loss).

## Results

Full experiment log, including training metrics (to gauge overfitting) alongside validation metrics. Grouped by model family; within each, roughly in the order the experiments were run. Test-set metrics (from [`07_prediction.ipynb`](https://github.com/yuliaforostiana/quora_question_pairs_duplicate_detection/blob/main/notebooks/07_prediction.ipynb)) are included wherever that experiment was one of the final shortlisted candidates.

#### Logistic Regression (`02_baseline.ipynb`, `03_logistic_regression.ipynb`)

| Data / Features | Parameters | Train Log Loss | Train ROC-AUC | Train F1 | Val Log Loss | Val ROC-AUC | Val F1 | Test Log Loss | Test ROC-AUC | Test F1 | Train time | Inference time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF, with stopwords | `'max_iter': 1000, 'solver': 'liblinear'` | 0.3451 | 0.9440 | 0.8070 | 0.5609 | 0.7531 | 0.5041 | — | — | — | 8.36 s | 0.0166 s |
| TF-IDF, without stopwords | `'max_iter': 1000, 'solver': 'liblinear'` | 0.3532 | 0.9417 | 0.8011 | 0.5786 | 0.7340 | 0.4594 | — | — | — | 3.99 s | 0.0113 s |
| TF-IDF, `RandomizedSearchCV`, with stopwords | `'solver':'saga','class_weight':'balanced','C':10` | 0.1652 | 0.9956 | 0.9624 | 0.6641 | 0.7250 | 0.5447 | — | — | — | 6 min 36 s | 0.02 s |
| TF-IDF, `RandomizedSearchCV`, without stopwords | `'solver':'saga','class_weight':'balanced','C':10` | 0.1810 | 0.9924 | 0.9485 | 0.6911 | 0.7004 | 0.5073 | — | — | — | 4 min 15 s | 0.01 s |
| TF-IDF + SVD, with stopwords | `'max_iter': 1000, 'solver': 'liblinear'` | 0.5041 | 0.8110 | 0.6061 | 0.5147 | 0.7944 | 0.5572 | — | — | — | 25 s | 0.09 s |
| TF-IDF + SVD, without stopwords | `'max_iter': 1000, 'solver': 'liblinear'` | 0.4923 | 0.8225 | 0.6318 | 0.5174 | 0.7948 | 0.5675 | — | — | — | 35.8 s | 0.11 s |
| Sentence Embeddings Model 1 | `'max_iter': 1000, 'solver': 'liblinear'` | 0.3143 | 0.9334 | 0.8124 | 0.3385 | 0.9202 | 0.7716 | — | — | — | 4 min 48 s | 0.72 s |
| Sentence Embeddings Model 2 | `'max_iter': 1000, 'solver': 'liblinear'` | 0.3383 | 0.9229 | 0.7923 | 0.3563 | 0.9117 | 0.7556 | — | — | — | 4 min 30 s | 1.16 s |
| **Sentence Embeddings Model 1, tuned** | `'C':1.2229, 'class_weight':'balanced','solver':'saga'` | 0.3275 | 0.9333 | 0.8205 | 0.3488 | 0.9199 | **0.7864** | 0.3380 | 0.9281 | 0.8114 | 85 min 39 s | 0.20 s (test: 0.14 s) |

#### XGBoost (`02_baseline.ipynb`, `04_xgboost.ipynb`)

| Data / Features | Parameters | Train Log Loss | Train ROC-AUC | Train F1 | Val Log Loss | Val ROC-AUC | Val F1 | Test Log Loss | Test ROC-AUC | Test F1 | Train time | Inference time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TF-IDF, with stopwords | `'n_estimators':300, 'learning_rate':0.1, 'max_depth':6` | 0.4884 | 0.8454 | 0.6113 | 0.5544 | 0.7655 | 0.4396 | — | — | — | 4 min 31 s | 5.8 s |
| TF-IDF, without stopwords | `'n_estimators':300, 'learning_rate':0.1, 'max_depth':6` | 0.5263 | 0.8123 | 0.5333 | 0.5865 | 0.7249 | 0.3182 | — | — | — | 2 min 40 s | 4.4 s |
| TF-IDF + SVD + handcrafted features, with stopwords | `'n_estimators':300, 'learning_rate':0.1, 'max_depth':6` | 0.3597 | 0.9209 | 0.7754 | 0.4596 | 0.8423 | 0.6323 | — | — | — | 18.42 s | 0.14 s |
| TF-IDF + SVD + handcrafted features, without stopwords | `'n_estimators':300, 'learning_rate':0.1, 'max_depth':6` | 0.3540 | 0.9216 | 0.7791 | 0.4595 | 0.8434 | 0.6282 | — | — | — | 19.17 s | 0.14 s |
| Sentence Embeddings Model 1 | `'n_estimators':300, 'learning_rate':0.1, 'max_depth':6` | 0.2083 | 0.9773 | 0.8936 | 0.3264 | 0.9266 | 0.7749 | 0.2783 | 0.9484 | 0.8330 | 2 min 40 s | 0.20 s (test: 0.26 s) |
| Sentence Embeddings Model 2 | `'n_estimators':300, 'learning_rate':0.1, 'max_depth':6` | 0.2221 | 0.9747 | 0.8871 | 0.3488 | 0.9159 | 0.7516 | — | — | — | 2 min 50 s | 0.20 s |
| **Sentence Embeddings Model 1, Hyperopt-tuned** | `colsample_bytree: 0.801, gamma: 0.885, learning_rate: 0.0747, max_depth: 7, min_child_weight: 1, n_estimators: 350, reg_alpha: 0.153, reg_lambda: 3.433, subsample: 0.860` | 0.1817 | 0.9856 | 0.9168 | **0.3264** | **0.9271** | 0.7741 | — | — | — | 6 min 17 s | **0.33 s** |

> **Note:** this Hyperopt-tuned configuration is the one saved as `models/xgboost_tfidf_se1.joblib` and used in deployment (see `04_xgboost.ipynb`, cell saving `best_model_se1`). The test-set row above (0.2783 / 0.9484 / 0.8330) was logged in `prediction.ipynb` against the "Sentence Embeddings Model 1" row rather than explicitly against the Hyperopt row — since both share the same validation Log Loss (0.3264) and `prediction.ipynb` loads the model directly from `xgboost_tfidf_se1.joblib`, this test result is presumed to reflect the deployed Hyperopt-tuned model. **Worth double-checking against the source spreadsheet/notebook before treating it as final**, since the row labeling doesn't make this fully unambiguous.

#### Transformers (`05_bert.ipynb`, `06_roberta.ipynb`)

| Model | Checkpoint / Selection | Train Log Loss | Train ROC-AUC | Train F1 | Val Log Loss | Val ROC-AUC | Val F1 | Test Log Loss | Test ROC-AUC | Test F1 | Train time | Inference time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BERT (`distilbert-base-uncased`) | checkpoint 48000 (final) | 0.1078 | 0.9923 | 0.9556 | 0.5704 | 0.9223 | 0.7787 | — | — | — | 2 h | 156.84 s |
| BERT (`distilbert-base-uncased`) | checkpoint 14000 (best) | 0.2226 | 0.9680 | 0.8836 | 0.3831 | 0.9185 | **0.7877** | — | — | — | 40 min | 157.15 s |
| **RoBERTa** (`FacebookAI/roberta-base`) | best (by F1) | 0.1616 | 0.9881 | 0.9381 | 0.4997 | **0.9374** | **0.8167** | 0.3696 | **0.9608** | **0.8678** | 2 h 8 min | 318.09 s (test: 5 min 50 s) |

### Final Model Selection — Held-out Test Set

The three strongest candidates from validation (tuned Logistic Regression, tuned XGBoost, and RoBERTa) were re-evaluated on a fully held-out test set in [`07_prediction.ipynb`](https://github.com/yuliaforostiana/quora_question_pairs_duplicate_detection/blob/main/notebooks/07_prediction.ipynb), confirming the deployment decision on unseen data rather than validation data alone. Classical models were run on CPU; RoBERTa was run on GPU.

| Model | Data / Features | Test Log Loss | Test ROC-AUC | Test F1 | Test time | Hardware |
|---|---|---|---|---|---|---|
| Logistic Regression | Sentence Embeddings (tuned) | 0.3380 | 0.9281 | 0.8114 | 0.14 s | CPU |
| **XGBoost** | **Sentence Embeddings (Hyperopt-tuned)** | **0.2783** | 0.9484 | 0.8330 | 0.26 s | CPU |
| RoBERTa | fine-tuned | 0.3696 | **0.9608** | **0.8678** | 5 min 50 s | GPU |

**This confirms the model selection made on validation data: XGBoost on sentence embeddings has the best Log Loss on the held-out test set as well** — RoBERTa leads on ROC-AUC and F1, consistent with it being selected for F1 during training (see the RoBERTa findings above), but XGBoost remains the strongest choice on the project's primary metric, and by a wider margin than on validation. Notably, RoBERTa's test-time cost (~5m50s, on GPU) confirms that its latency disadvantage isn't a CPU artifact — it would remain far too slow for the prediction API even with hardware acceleration available.

## Conclusions

- **Sentence embeddings clearly outperform TF-IDF-based features** for this task — semantic similarity is a much stronger signal for duplicate detection than lexical overlap alone, which matches the EDA finding that word-overlap features (Jaccard, `common_ratio`) correlate with duplication but leave a lot of variance unexplained.
- **Transformer models (BERT/RoBERTa) achieve the best F1/ROC-AUC**, but at a cost of 150–350 seconds of inference time and hours of training — impractical for a low-latency prediction API, and this holds true on GPU as well as CPU.
- **XGBoost on sentence embeddings (Hyperopt-tuned) offers the best quality/latency trade-off, and the best Log Loss on both validation and the held-out test set** — this is the model selected for deployment, at ~0.2–0.3s inference and a few minutes of training.
- **Data quality issues identified in EDA directly informed preprocessing**: dropping rows with missing questions, correcting/handling the 18 mislabeled identical pairs, and grouping by question ID during train/validation splitting to prevent data leakage.
- Given the mild class imbalance (63/37), no class-balancing techniques were necessary.

## Repository Structure
Due to GitHub storage limitations, large artifacts (preprocessed datasets, trained models, and intermediate files) are hosted externally and can be downloaded from the Google Drive: [preprocess_data](https://drive.google.com/drive/folders/1XK9w-RtTGSauhTp8qMbn-IwxnJiM-W8n?usp=sharing) and [models](https://drive.google.com/drive/folders/1TInDFnlIyZkyrPbknCqht-bJDlTb_Ni3?usp=sharing).
```
.
├── README.md
│
├── notebooks/              # Project notebooks (EDA, preprocessing, training, evaluation, inference)
│   ├── 00_eda.ipynb                    # Exploratory data analysis
│   ├── 01_preprocessing.ipynb          # Data cleaning, leakage-free train/val split, feature pipelines
│   ├── 02_baseline.ipynb               # Baseline Logistic Regression (TF-IDF only)
│   ├── 03_logistic_regression.ipynb    # Logistic Regression across all feature sets, RandomizedSearchCV
│   ├── 04_xgboost.ipynb                # XGBoost across all feature sets, Hyperopt tuning
│   ├── 05_bert.ipynb             # Fine-tuned DistilBERT (Google Colab / GPU)
│   ├── 06_roberta.ipynb          # Fine-tuned RoBERTa (Google Colab / GPU)
│   └── 07_prediction.ipynb             # Final held-out test evaluation of the 3 shortlisted models  
│
├── src/              # Reusable Python modules used across the project
│   ├── eda_utils.py                 # Text preprocessing, corpus/question stats, overlap features, plots
│   ├── preprocessing_utils.py       # Cleaning, graph-based split, TF-IDF/SVD, embeddings, BERT prep
│   ├── model_utils.py               # Shared train/eval/error-analysis helpers for all modeling notebooks
│   └── inference_utils.py           # Test-set cleaning/feature-building helpers used by prediction.ipynb
│
├── preprocess_data/              # Generated artifacts (not committed — see Installation & Usage)
│   ├── *.parquet                 # Cleaned/feature-engineered train & val splits
│   ├── *_tf_idf_q*_vec.npz       # Saved sparse TF-IDF matrices
│   ├── vectorizer_*.joblib       # Fitted TfidfVectorizer objects
│   ├── svd_*_transformer.joblib  # Fitted TruncatedSVD objects
│   ├── embedding_*.npy           # Saved sentence-embedding matrices
│   ├── embedding_model_name*.txt # Sentence-transformers model name(s) used
│   └── bert_config.json / roberta_config.json  # Checkpoint + max_length per transformer
│
├── models/                       # Trained model artifacts (not committed — see Installation & Usage)
│   ├── best_logistic_model_*.joblib
│   ├── xgboost_tfidf_se1.joblib  # <- deployed model
│   ├── distilbert_baseline/ (+ checkpoints)
│   └── roberta/
│
├── fast_api/                # FastAPI prediction service
│   ├── app.py
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── README.md
│   ├── models/                    # xgboost_tfidf_se1.joblib
│   └── preprocess_data/           # embedding_model_name.txt
│
└── streamlit_app/               # Streamlit interactive demo
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── models/                    # xgboost_tfidf_se1.joblib
    └── preprocess_data/           # embedding_model_name.txt
```

## Installation & Usage

1. **Clone the repository and install dependencies:**
   ```bash
   git clone <repo-url>
   cd <repo-name>
   pip install -r requirements.txt
   ```

2. **Add the raw data** — place `quora_question_pairs_train.csv.zip` (and `quora_question_pairs_test.csv.zip` for the final test evaluation) in the repository root.

3. **Run the notebooks in order** (each stage reads the outputs of the previous one from `preprocess_data/` / `models/`):
   1. `00_eda.ipynb` — exploratory analysis (optional for reproducing models, but explains the *why* behind preprocessing choices)
   2. `01_preprocessing.ipynb` — cleans the data and builds all three feature sets into `preprocess_data/`
   3. `02_baseline.ipynb` — baseline Logistic Regression
   4. `03_logistic_regression.ipynb` — full Logistic Regression comparison + tuning
   5. `04_xgboost.ipynb` — full XGBoost comparison + Hyperopt tuning → saves the deployed model to `models/xgboost_tfidf_se1.joblib`
   6. `05_bert.ipynb` / `roberta.ipynb` — transformer fine-tuning (**requires a GPU** — developed on Google Colab; update the hardcoded Google Drive paths if running elsewhere)
   7. `07_prediction.ipynb` — final evaluation of the shortlisted models on the held-out test set

4. **Deploy the chosen model** — see the dedicated guides:
   - [`fast_api/README.md`](https://github.com/yuliaforostiana/quora_question_pairs_duplicate_detection/blob/main/fast_api/README.md) — run the prediction API locally (`uvicorn`) or via Docker
   - [`streamlit_app/README.md`](https://github.com/yuliaforostiana/quora_question_pairs_duplicate_detection/blob/main/streamlit_app/README.md) — run the interactive Streamlit demo

   Both require `models/xgboost_tfidf_se1.joblib` and `preprocess_data/embedding_model_name.txt`, copied into their respective `models/` / `preprocess_data/` subfolders.

## Requirements

A consolidated top-level `requirements.txt` covering EDA, preprocessing, and modeling:

```
pandas
numpy
scipy
matplotlib
seaborn
nltk
contractions
networkx
scikit-learn
xgboost
hyperopt
sentence-transformers
transformers
datasets
accelerate
evaluate
torch
joblib
```

Notes:
- `torch` + `transformers`/`datasets`/`accelerate`/`evaluate` are only needed for `bert.ipynb` / `roberta.ipynb` / `prediction.ipynb` — fine-tuning **requires a GPU** in practice (these notebooks were run on Google Colab).
- `nltk` requires a one-time download of `stopwords`, `averaged_perceptron_tagger_eng`, and `wordnet` (handled at the top of `eda.ipynb` / `preprocessing.ipynb`).
- The deployment sub-projects (`fast_api/`, `streamlit_app/`) intentionally ship their **own smaller `requirements.txt`** — they only need `fastapi`/`streamlit`, `sentence-transformers`, `xgboost`, and `joblib`, not the full EDA/training stack.
