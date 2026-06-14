# IMDB Sentiment Analysis — Hybrid Learning Pipeline

A machine learning pipeline for binary sentiment classification on the IMDB movie review dataset, combining **unsupervised clustering** (PCA + K-Means) with **supervised learning** (Logistic Regression) in a hybrid pseudo-labeling approach.

---

## Overview

This project explores a hybrid learning strategy where:
1. Reviews are preprocessed and vectorized using TF-IDF
2. Dimensionality is reduced via PCA
3. K-Means clustering generates **pseudo-labels** (no ground truth used during training)
4. A Logistic Regression classifier is trained on these pseudo-labels
5. The hybrid model is evaluated against a fully supervised baseline

The goal is to assess how well an unsupervised approach can approximate supervised sentiment classification.

---

## Project Structure

```
.
├── datasets/
│   └── IMDB Dataset.csv          # Raw dataset (50,000 reviews)
├── notebooks/
│   ├── outputs/                  # Intermediate outputs from notebooks
│   ├── 01_preprocessing.ipynb    # Data loading & text preprocessing
│   ├── 02_pca_kmeans.ipynb       # PCA dimensionality reduction & K-Means clustering
│   ├── 03_supervised.ipynb       # Supervised learning with pseudo-labels
│   └── 04_evaluation.ipynb       # Model evaluation & comparison
├── outputs/
│   └── figures/
│       ├── confusion_matrices.png
│       ├── pca_variance.png
│       └── performance_comparison.png
├── src/
│   ├── clustering.py             # PCA & K-Means logic
│   ├── evaluation.py             # Metrics & comparison utilities
│   ├── preprocess.py             # Text cleaning & normalization
│   └── vectorize.py              # TF-IDF vectorization
├── run_notebooks.py              # Execute all notebooks sequentially
├── run_pipeline.py               # Run the full pipeline as a Python script
├── requirements.txt
└── README.md
```

---

## Dataset

**IMDB Movie Reviews Dataset** — 50,000 reviews, perfectly balanced:
- 25,000 positive reviews
- 25,000 negative reviews

Source: [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

Place the dataset at `datasets/IMDB Dataset.csv` before running the pipeline.

---

## Pipeline Phases

### Phase 1 — Preprocessing (`preprocess.py`)
- Strip HTML tags and URLs
- Lowercase and remove special characters
- Tokenize using NLTK (`word_tokenize`)
- Remove standard stopwords while **preserving negation words** (`not`, `no`, `never`, `nor`)
- Remove domain-specific custom stopwords: `movie`, `film`, `character`, `story`, `watch`, `time`
- Lemmatize tokens using `WordNetLemmatizer`

> Average review length reduced from **231 words → 112 words** (~51.5% reduction)

### Phase 2 — Vectorization (`vectorize.py`)

TF-IDF parameters:

| Parameter | Value | Reason |
|---|---|---|
| `max_features` | 10,000 | Caps vocabulary size for efficiency |
| `ngram_range` | (1, 2) | Captures bigram context (e.g. "not good") |
| `min_df` | 5 | Filters rare/typo terms |
| `dtype` | float32 | Reduces memory vs float64 |

Result: sparse matrix of shape **50,000 × 10,000** (sparsity: 0.9912)

### Phase 3 — PCA (`clustering.py`)
- Experiments with `n_components ∈ {20, 50, 100, 200}`
- Even at 200 components, only ~16.3% of variance is retained — reflecting the high complexity of TF-IDF space

| n_components | Explained Variance |
|---|---|
| 20 | 3.47% |
| 50 | 6.48% |
| 100 | 10.32% |
| 200 | 16.30% |

### Phase 4 — K-Means Clustering (`clustering.py`)
- `n_clusters=2` (positive / negative)
- Best silhouette score achieved at **20 components**

| n_components | Silhouette Score |
|---|---|
| 20 | 0.0896 ✓ best |
| 50 | 0.0581 |
| 100 | 0.0420 |
| 200 | 0.0295 |

Cluster keyword analysis (from centroid TF-IDF scores):
- **Cluster 0** — `great, performance, love, excellent, family, wonderful, best` → **positive**
- **Cluster 1** — `bad, dont, not, no, worst, thing, acting` → **negative**

Pseudo-label alignment accuracy after flipping: **70.81%**

### Phase 5 — Supervised Learning
- Logistic Regression trained on K-Means pseudo-labels
- `GridSearchCV` with 5-fold cross-validation over `C ∈ {0.1, 1, 10}`
- Best parameter: **C = 10** (CV F1: 0.9966)
- 80/20 train-test split

### Phase 6 — Evaluation (`evaluation.py`)
- Metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrices for both hybrid and direct supervised models
- Performance gap analysis between the two approaches

---

## Results

### Hybrid vs. Direct Supervised Learning

| Metric | Hybrid Learning | Direct Supervised | Gap |
|---|---|---|---|
| Accuracy | 0.4957 | 0.8345 | 0.3388 |
| Precision | 0.4955 | 0.8347 | 0.3392 |
| Recall | 0.4957 | 0.8345 | 0.3388 |
| F1-Score | 0.4898 | 0.8345 | 0.3447 |

The hybrid model's ~49.6% accuracy is close to random chance, demonstrating that pseudo-label quality is the primary bottleneck in this pipeline. The direct supervised baseline achieves **83.45% accuracy** on the same test set.

### Key Failure Cases

1. **Sarcasm** — "This movie was so good that I almost fell asleep." The word "good" fires a positive signal, but the actual sentiment is negative. TF-IDF has no way to model this.
2. **Negation** — "The acting was not bad." The word "bad" carries high negative weight even though the phrase is positive. Bigrams help but don't fully solve this.
3. **PCA information loss** — Dimensionality reduction discards fine-grained sentiment signals, causing semantically different reviews to map to nearly identical PCA representations.
4. **Cluster overlap** — Mixed-sentiment reviews like "The story was boring, but the acting was excellent" are inherently ambiguous and hard to cluster correctly.
5. **Error propagation** — Noise in the pseudo-labels is inherited by the Logistic Regression model, compounding inaccuracies during training.

---

## Getting Started

### 1. Clone the repository

```bash
git clone <repo-url>
cd <repo-folder>
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Add the dataset

Download the IMDB dataset and place it at:
```
datasets/IMDB Dataset.csv
```

### 4. Run the pipeline

**Option A — Run as a Python script (recommended for speed):**
```bash
python run_pipeline.py
```

**Option B — Run as Jupyter notebooks:**
```bash
python run_notebooks.py
```
Or open and run notebooks manually in order: `01` → `02` → `03` → `04`.

---

## Requirements

Key dependencies (see `requirements.txt` for full list):

- `pandas`, `numpy`
- `scikit-learn`
- `nltk`
- `matplotlib`, `seaborn`
- `jupyter`, `nbconvert`

On first run, NLTK will automatically download required corpora (`punkt`, `stopwords`, `wordnet`).

---

## Outputs

After running the pipeline, the following files are generated:

| File | Description |
|---|---|
| `outputs/cleaned_reviews.csv` | Preprocessed reviews with labels |
| `outputs/real_labels.npy` | Ground-truth binary labels |
| `outputs/best_X_pca.npy` | PCA-reduced feature matrix (20 components) |
| `outputs/best_cluster_labels.npy` | Aligned K-Means cluster labels |
| `outputs/y_pred_test.npy` | Hybrid model predictions on test set |
| `outputs/evaluation_metrics.csv` | Metric comparison table |
| `outputs/error_analysis.csv` | Sample misclassification cases |
| `outputs/figures/pca_variance.png` | Cumulative explained variance plots |
| `outputs/figures/confusion_matrices.png` | Confusion matrices for both models |
| `outputs/figures/performance_comparison.png` | Side-by-side metric comparison |

---

## Methodology Notes

**Why pseudo-labeling?**
Pseudo-labeling simulates a real-world scenario where labeled data is unavailable or expensive to obtain. K-Means on TF-IDF + PCA features can discover rudimentary sentiment structure without any ground truth, at the cost of noisy labels.

**Cluster alignment:**
K-Means cluster IDs (0 and 1) are arbitrary. The pipeline checks both normal and flipped assignments against real labels to pick the best mapping before training.

**Evaluation design:**
The hybrid model is evaluated against **ground-truth labels** (not pseudo-labels) to give an honest measure of real-world performance. A direct supervised baseline (trained on real labels) is included for comparison.

**Why the gap is large:**
TF-IDF + K-Means cannot model semantic context — negation, sarcasm, and mixed-sentiment reviews all confuse the clustering step. The resulting pseudo-labels (~70.8% aligned) are too noisy to train an effective classifier.

---

## Future Work

- Replace TF-IDF with word embeddings (Word2Vec, FastText, GloVe)
- Use transformer-based models (BERT, RoBERTa) for context-aware representations
- Try density-based clustering (DBSCAN) instead of K-Means
- Explore dimensionality reduction alternatives (UMAP, t-SNE)
- Apply semi-supervised learning techniques to better leverage unlabeled data

---

## License

This project is for academic purposes (IF-605 Machine Learning).
