# CSE440 NLP Project — News Topic Classification

## Overview

An end-to-end Natural Language Processing pipeline for multi-class news headline classification. The project classifies news headlines into topic categories using a full ML workflow: exploratory data analysis, three text preprocessing strategies, two text representation methods (TF-IDF and Skip-gram embeddings), and five model architectures (Logistic Regression, DNN, SimpleRNN, GRU, LSTM). Models are systematically compared and the best configuration is evaluated on a held-out test set.

**Dataset:** 96,646 training headlines and 12,000 test headlines across 4 news topic classes (`News Headline` → `News Topic`)

---

## Project Structure

```
cse440-NLP-project.ipynb
data/
  Training_data_13.csv
  Test_data.csv
```

---

## Pipeline Stages

### 1. Environment Setup & Imports
- Installed and imported: `beautifulsoup4`, `wordcloud`, `gensim`, `nltk`, `tensorflow`, `sklearn`, `scipy`
- Set global random seeds (`SEED=42`) across Python, NumPy, and TensorFlow for full reproducibility
- Configured caching utilities (`save_pickle` / `load_pickle`) to save intermediate results and avoid costly recomputation between runs

---

### 2. Exploratory Data Analysis (EDA)

A thorough data investigation before any modeling:

- **Dataset overview:** shape, missing values, unique texts, duplicate rows, average word and character counts
- **Class distribution:** table and grouped bar chart comparing train vs. test label frequencies — revealed class imbalance across the 4 topics
- **Raw sample inspection:** spot-checked random headline samples
- **HTML artifact cleaning:** decoded HTML entities (`&amp;`, `&lt;`, `#39;`, etc.) using `BeautifulSoup`
- **Length statistics by class:** mean/median word count and character count per topic using `groupby`
- **Word-count histograms and boxplots:** visualized headline length distributions across all classes
- **Source-tag analysis:** measured Reuters, AP, AFP tag frequency per class; found source tags vary significantly by topic (potential data leakage signal)
- **Top unigrams and bigrams per class:** computed after stripping source tags to expose genuine topic-specific vocabulary
- **Word clouds per class:** generated visual frequency maps to intuitively confirm class-specific language

**EDA findings:** No missing values; dataset has exact duplicate headlines that require duplicate-aware splitting; "Sports" headlines tend to be shorter; source tags correlate with topic distribution.

---

### 3. Text Preprocessing — Three Variants Compared

Three preprocessing strategies were built and saved as separate CSVs for reproducible comparison:

| Variant | Steps Applied |
|---|---|
| **None** | Raw text — no changes |
| **Extreme** | HTML decode → source tag removal → lowercase → punctuation removal → stopword removal → Porter stemming |
| **Optimum** | HTML decode → source tag removal → lowercase → light normalization (no stopword removal, no stemming) |

- Used `PorterStemmer` from `nltk` for the extreme variant
- Applied `sklearn`'s `ENGLISH_STOP_WORDS` for stopword filtering
- All three variants were fed into every downstream model to quantify the effect of preprocessing on accuracy

---

### 4. Duplicate-Aware Train / Validation Split

- Headlines in the training set contain many exact duplicates
- A naive random split would let the same headline appear in both train and val, inflating scores
- Solved by grouping on normalized raw text and splitting at group level using `LabelEncoder` + hash keys
- Ensures the validation set measures true generalization, not memorization

---

### 5. Model 1 — TF-IDF + Logistic Regression (Baseline)

- **TF-IDF:** `max_features=50,000`, `ngram_range=(1,2)`, `min_df=3`, `max_df=0.95`
- **Logistic Regression:** `class_weight="balanced"`, hyperparameter search over `C=[1.0, 3.0, 5.0]` and `max_iter=[500, 1500]`
- Run on all three preprocessing variants
- Logged accuracy and macro F1 for each combination
- Plotted comparative bar charts of results across preprocessing variants

---

### 6. Model 2 — TF-IDF + Deep Neural Network (DNN)

Architecture:
```
Dense(512, relu) → Dropout → Dense(256, relu) → Dropout → Dense(128, relu) → Dense(n_classes, softmax)
```
- Input: TF-IDF sparse vectors (optionally reduced via `TruncatedSVD`)
- Callbacks used: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`, `CSVLogger`
- Run on all three preprocessing variants and logged to the experiment tracker

---

### 7. Skip-gram Word Embeddings

- Trained a custom `Word2Vec` model using Skip-gram (`sg=1`): `vector_size=200`, `window=5`, `min_count=2`, `epochs=15`
- Built a Keras-compatible embedding matrix from Skip-gram weights
- Tokenized with Keras `Tokenizer` (`max_vocab=50,000`) and padded sequences to `max_len=60`
- Embedding matrix injected into recurrent models as frozen (pretrained) weights

---

### 8. Models 3–5 — Recurrent Neural Networks (RNN, GRU, LSTM)

Three sequence model architectures, each using the Skip-gram embedding matrix:

| Model | Architecture |
|---|---|
| **SimpleRNN** | Embedding → SpatialDropout1D → SimpleRNN(128) → Dropout → Dense(softmax) |
| **GRU** | Embedding → SpatialDropout1D → Bidirectional(GRU(128)) → Dropout → Dense(softmax) |
| **LSTM** | Embedding → SpatialDropout1D → Bidirectional(LSTM(128)) → Dropout → Dense(softmax) |

- All compiled with `Adam` optimizer and `sparse_categorical_crossentropy` (integer labels, no one-hot needed)
- Hyperparameter grid search: `dropout_rate=[0.3, 0.5]`, `batch_size=[32, 64]`, `epochs=[5, 8]`
- Model checkpoints saved for best validation accuracy; reloaded after training
- Run on all three preprocessing variants — 9 run combinations per architecture

---

### 9. Experiment Comparison & Final Evaluation

- All runs logged to `EXPERIMENT_LOG` with model name, preprocessing, representation, accuracy, and macro F1
- Sorted results table and bar chart comparing all models
- **Best configuration identified:** `preprocessing=optimum`, `representation=Skip-gram`, `model=Bidirectional GRU`
- Final model retrained on the **full training set** (no held-out val) and evaluated on the test set
- Final output: accuracy, macro F1, full `classification_report`, and `confusion_matrix`

---

## Skills Obtained

| Skill | Tools / Libraries |
|---|---|
| End-to-end NLP pipeline design | Full project, all stages |
| Exploratory data analysis for text | `pandas`, `matplotlib`, `Counter`, `BeautifulSoup` |
| HTML entity decoding and text normalization | `BeautifulSoup`, `re`, `html` |
| Word frequency analysis (unigrams, bigrams) | `CountVectorizer`, `Counter` |
| Word cloud visualization | `WordCloud` |
| Three-way preprocessing comparison | Custom `preprocess_none`, `extreme`, `optimum` functions |
| Text stemming and stopword removal | `nltk.PorterStemmer`, `sklearn.ENGLISH_STOP_WORDS` |
| Duplicate-aware train/val splitting | Custom group-key splitting with `LabelEncoder` |
| TF-IDF feature extraction with n-grams | `TfidfVectorizer` with bigrams |
| Dimensionality reduction on sparse vectors | `TruncatedSVD` |
| Logistic Regression with class balancing | `sklearn.LogisticRegression` (`class_weight="balanced"`) |
| Deep neural network for text (DNN) | Keras `Sequential`, `Dense`, `Dropout` |
| Custom Word2Vec Skip-gram embeddings | `gensim.models.Word2Vec` (`sg=1`) |
| Sequence modeling for NLP | Keras `SimpleRNN`, `GRU`, `LSTM` |
| Bidirectional recurrent layers | `Bidirectional(GRU)`, `Bidirectional(LSTM)` |
| Embedding layers with pretrained weights | `layers.Embedding` with Skip-gram weight matrix |
| Spatial dropout for sequence regularization | `SpatialDropout1D` |
| Training callbacks | `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`, `CSVLogger` |
| Hyperparameter grid search (manual) | Custom loop over `units`, `dropout_rate`, `batch_size`, `epochs` |
| Model evaluation and comparison | `classification_report`, `confusion_matrix`, macro F1 |
| Experiment logging and result tracking | Custom `EXPERIMENT_LOG` with pandas DataFrame output |
| Reproducibility engineering | Seeds, caching (pickle), checkpoint reload |
| Cloud notebook development | Google Colab, file I/O with `pathlib` |

---

## Key Concepts Learned

- **Preprocessing matters:** aggressive stemming and stopword removal (`extreme`) can actually *hurt* performance on short headlines where every word carries signal — `optimum` (light cleaning) outperformed `extreme`
- **TF-IDF vs. learned embeddings:** TF-IDF with Logistic Regression is a strong, fast baseline; Skip-gram embeddings capture semantic meaning that TF-IDF misses, giving recurrent models an advantage
- **Skip-gram vs. CBOW:** Skip-gram (predicts context from a word) works better on smaller or imbalanced vocabularies — well-suited for news headlines
- **Bidirectional RNNs:** reading a sequence both forward and backward gives GRU/LSTM access to full context at every position, which improves classification of short texts
- **SpatialDropout1D:** drops entire feature maps (timesteps) rather than individual units — more effective than regular dropout for sequence models with embeddings
- **`sparse_categorical_crossentropy` vs. `categorical_crossentropy`:** use sparse when labels are integers; use categorical when labels are one-hot — avoids unnecessary memory overhead
- **Duplicate-aware splits:** failing to account for duplicates inflates validation accuracy and masks overfitting
- **`class_weight="balanced"`:** automatically adjusts the loss function to penalize errors on minority classes — essential for imbalanced classification
- **ReduceLROnPlateau:** halves the learning rate when validation loss plateaus, recovering training that would otherwise stagnate
- **ModelCheckpoint + restore:** saves the best model weights mid-training so the final model isn't degraded by later epochs that overfit

---

## Environment

- Python 3
- Google Colab
- Libraries: `tensorflow`, `keras`, `gensim`, `sklearn`, `nltk`, `numpy`, `pandas`, `matplotlib`, `wordcloud`, `beautifulsoup4`, `scipy`
