
# CSE440 Lab 2 — Word Representations

## Overview

This lab explores core methods for representing words as numerical vectors in Natural Language Processing (NLP). Starting from simple count-based approaches and progressing to context-aware neural embeddings, it covers both the theory and practical implementation of each technique using Python.

---

## Topics Covered

### 1. Bag of Words (BoW)
- Converted raw text into fixed-length numeric vectors using word occurrence counts
- Built a BoW model with `sklearn`'s `CountVectorizer`
- Constructed and interpreted a term-document matrix
- Applied the model to a real-world corpus of 50,000 text samples

**Key insight:** BoW captures word frequency but ignores word order and context.

---

### 2. TF-IDF (Term Frequency–Inverse Document Frequency)
- Applied `TfidfVectorizer` to score words by their importance relative to the entire corpus
- Identified the top 15 most significant terms across 50,000 documents
- Understood how common words (e.g., "the", "and") are down-weighted by the IDF component

**Key insight:** TF-IDF balances local frequency (TF) against global commonness (IDF) to surface the most meaningful terms in a document.

---

### 3. GloVe (Global Vectors for Word Representation)
- Loaded pretrained GloVe embeddings (6B tokens, 100D)
- Retrieved dense vector representations for individual words
- Performed word analogy arithmetic: `king − man + woman ≈ queen`
- Computed cosine similarity between word vectors to measure semantic closeness

**Key insight:** GloVe captures global co-occurrence statistics, enabling semantic relationships to be expressed through simple vector arithmetic.

---

### 4. Word2Vec (CBOW & Skip-gram)
- Trained custom Word2Vec models on the Reuters corpus using `gensim`
- Compared two training architectures:
  - **CBOW** (`sg=0`) — predicts a target word from its surrounding context words
  - **Skip-gram** (`sg=1`) — predicts context words from a single target word
- Tuned model hyperparameters: `vector_size`, `window`, `min_count`, `epochs`
- Performed word analogy tasks: `oil − energy + wheat`
- Evaluated trained embeddings using cosine similarity

**Key insight:** Skip-gram performs better for rare words; CBOW is faster and scales better with larger corpora.

---

### 5. Cosine Similarity
- Measured semantic similarity between word vectors from both Word2Vec and GloVe models
- Interpreted the similarity scale: `1` = identical direction, `0` = unrelated, `−1` = opposite
- Used cosine similarity as the scoring function for nearest-neighbor word analogy search

---

## Skills Obtained

| Skill | Tools / Libraries |
|---|---|
| Text vectorization | `sklearn.CountVectorizer`, `TfidfVectorizer` |
| Loading pretrained embeddings | NumPy file parsing (GloVe `.txt` format) |
| Training word embeddings from scratch | `gensim.models.Word2Vec` |
| Semantic similarity computation | `sklearn.metrics.pairwise.cosine_similarity` |
| Corpus preprocessing | Tokenization, lowercasing, punctuation removal |
| NLP data handling | `pandas`, `numpy`, `nltk` |
| Cloud notebook development | Google Colab + Google Drive integration |

---

## Files

- [cse440_lab2_spring26.ipynb](cse440_lab2_spring26.ipynb) — Lab notebook with all implementations and outputs

---

## Environment

- Python 3
- Google Colab
- Libraries: `sklearn`, `gensim`, `numpy`, `pandas`, `nltk`
