# CSE440 Lab 3 — Neural Networks for Classification

## Overview

This lab introduces deep learning for classification tasks using TensorFlow and Keras. It covers three progressively complex classification problems: tabular binary classification on medical data, text-based spam detection with word embeddings, and multiclass classification on the Iris dataset. Each task demonstrates a complete ML pipeline — from raw data to trained neural network — and benchmarks deep learning against classical ML models.

---

## Tasks

### Task 1 — Binary Classification on Medical Tabular Data (Cancer Dataset)

**Dataset:** Breast cancer dataset (CSV via Google Drive) — features describing cell nuclei, labeled benign (0) or malignant (1)

**What was done:**
- Loaded and explored the dataset: shape, column types, descriptive statistics, missing value audit
- Dropped non-informative ID column; separated feature matrix `X` and target `y`
- Split data 80/20 into train and test sets using `train_test_split`
- Standardized features using `StandardScaler` (fit on train, transform on test)
- Trained three classical ML baselines:
  - **Logistic Regression** — linear decision boundary
  - **Naive Bayes** (Gaussian) — probabilistic classifier
  - **Random Forest** — ensemble of decision trees
- Built a Keras `Sequential` neural network:
  - Input → `Dense(10, activation='relu')` → `Dense(1, activation='sigmoid')`
- Compiled with `Adam` optimizer and `binary_crossentropy` loss
- Trained for 10 epochs with 20% validation split
- Plotted training vs. validation accuracy and loss curves

**Key insight:** A shallow neural network can match or exceed classical ML models on structured tabular data. `sigmoid` output + binary cross-entropy is the standard pairing for binary classification.

---

### Task 2 — SMS Spam Detection (Text Classification with Embeddings)

**Dataset:** SMS Spam Collection — 5,572 messages labeled `ham` (legitimate) or `spam`

**What was done:**
- Loaded the dataset from a public GitHub URL; parsed tab-separated format
- Encoded string labels to integers using `LabelEncoder`
- Tokenized raw text with Keras `Tokenizer` (vocabulary capped at 5,000 words, OOV token added)
- Converted messages to integer sequences and padded/truncated to length 100 with `pad_sequences`
- Built a Keras `Sequential` model:
  - `Embedding(5000, output_dim=16, input_length=100)` — learns dense word vectors
  - `Flatten()` — unrolls the embedding matrix
  - `Dense(64, activation='relu')` → `Dense(1, activation='sigmoid')`
- Used `EarlyStopping(patience=3, restore_best_weights=True)` to prevent overfitting
- Evaluated with full `classification_report` (precision, recall, F1-score) and `confusion_matrix`
- Plotted training/validation accuracy and loss curves

**Key insight:** An `Embedding` layer lets the network learn its own word representations end-to-end, turning raw text into something a dense network can process. `EarlyStopping` automates regularization by halting training when validation loss stops improving.

---

### Task 3 — Multiclass Classification on Iris Dataset

**Dataset:** Iris — 150 samples, 4 numeric features (sepal/petal length & width), 3 flower species

**What was done:**
- Loaded `sklearn`'s built-in Iris dataset
- Standardized features with `StandardScaler`
- One-hot encoded integer labels using `tf.keras.utils.to_categorical` (required for categorical cross-entropy)
- Built a deeper Keras `Sequential` model:
  - `Dense(64, activation='relu')` → `Dense(32, activation='relu')` → `Dense(3, activation='softmax')`
- Compiled with `categorical_crossentropy` loss and `Adam` optimizer
- Trained for 30 epochs with batch size 16 and 20% validation split
- Used `np.argmax` to convert softmax probability vectors back to class indices
- Confirmed accuracy using `sklearn`'s `accuracy_score` alongside Keras `.evaluate()`
- Plotted training/validation accuracy and loss curves

**Key insight:** For multiclass problems, the output layer uses `softmax` (outputs a probability distribution over classes) paired with `categorical_crossentropy` loss — in contrast to the `sigmoid` + `binary_crossentropy` pair used for binary tasks.

---

## Skills Obtained

| Skill | Tools / Libraries |
|---|---|
| Building neural networks | `tensorflow`, `keras.Sequential`, `Dense`, `Embedding` layers |
| Binary & multiclass classification | `sigmoid` + binary CE vs. `softmax` + categorical CE |
| Text preprocessing pipeline | `Tokenizer`, `pad_sequences`, `LabelEncoder` |
| Word embedding layers | `layers.Embedding` (learned end-to-end) |
| Classical ML baselines | `LogisticRegression`, `GaussianNB`, `RandomForestClassifier` |
| Feature scaling | `StandardScaler` (fit/transform pattern) |
| One-hot encoding for labels | `tf.keras.utils.to_categorical` |
| Preventing overfitting | `EarlyStopping` callback with `restore_best_weights` |
| Model evaluation | `accuracy_score`, `classification_report`, `confusion_matrix` |
| Training diagnostics | Plotting accuracy/loss curves with `matplotlib` |
| Dataset handling | `pandas`, `numpy`, `sklearn.datasets.load_iris` |
| Cloud notebook development | Google Colab + Google Drive integration |

---

## Key Concepts Learned

- **Neural network architecture choices:** how depth, width, and activation functions change what a model can learn
- **Binary vs. multiclass output layers:** `sigmoid`/binary CE for 2 classes vs. `softmax`/categorical CE for N classes
- **Embedding layers:** replacing hand-crafted text features with trainable dense vector representations learned from data
- **The fit/transform pattern:** why `StandardScaler` must be fit on training data only, then applied to test data — to prevent data leakage
- **EarlyStopping:** automatic regularization that monitors validation loss and stops training when generalization stops improving
- **Validation splits:** using a held-out validation portion during training to track overfitting in real time
- **Classical ML vs. deep learning:** when to use simple, interpretable models (Logistic Regression, Random Forest) versus neural networks
- **argmax decoding:** converting softmax probability vectors into discrete class predictions

---

## Files

- [CSE440_Lab_03_Spring2026.ipynb](CSE440_Lab_03_Spring2026.ipynb) — Lab notebook with all implementations and outputs

---

## Environment

- Python 3
- Google Colab
- Libraries: `tensorflow`, `keras`, `sklearn`, `numpy`, `pandas`, `matplotlib`, `nltk`
