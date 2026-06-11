# CSE440 — Natural Language Processing | Lab 1

## Course Overview

CSE440 is an undergraduate Natural Language Processing (NLP) course. The course introduces the foundational techniques used to make computers understand, analyze, and derive meaning from human language. Lab 1 covers core text preprocessing and analysis workflows using Python's **NLTK** library, working with real-world corpora including financial news, presidential speeches, movie reviews, and multilingual human rights documents.

---

## What This Lab Does

The lab is structured around five hands-on questions, each building on a new NLP concept:

| # | Topic | Corpus Used |
|---|-------|-------------|
| 1 | Exploring NLTK corpora | Reuters (financial news), Movie Reviews |
| 2 | Tokenization, stopword removal, word cloud visualization | State of the Union addresses (Nixon 1972, Johnson 1965) |
| 3 | Lemmatization & frequency distribution | Movie Reviews (negative sentiment) |
| 4 | N-gram extraction & co-occurrence matrix | Reuters news article |
| 5 | Vowel trigram analysis | UDHR multilingual corpus (German) |

### Question 1 — Corpus Exploration
Loaded and listed categories from the **Reuters** corpus (90 financial topics like `crude`, `gold`, `interest`) and the **Movie Reviews** corpus (binary: `pos` / `neg`), getting familiar with NLTK's built-in datasets.

### Question 2 — Text Preprocessing & Visualization
Compared Nixon's 1972 and Johnson's 1965 State of the Union speeches by:
- Lowercasing and **word tokenizing** raw text
- Removing **stopwords** and non-alphabetic tokens
- Generating **word cloud** visualizations to reveal each president's key themes

### Question 3 — Lemmatization & Bar Chart
Analyzed a negative movie review by:
- Tokenizing and filtering stopwords
- Applying **WordNet lemmatization** (verb form) to normalize words (e.g., *running* → *run*)
- Plotting the **top 20 most frequent words** as a bar chart using Matplotlib

### Question 4 — Bigrams & Co-occurrence Matrix
Processed a Reuters financial article to:
- Extract the **top 12 most frequent words**
- Generate all **bigrams** (consecutive word pairs)
- Build a **co-occurrence matrix** (NumPy) showing how often top words appear side-by-side

### Question 5 — Vowel Trigrams
Worked with German text from the UDHR multilingual corpus to:
- Extract only vowel characters from each word
- Construct **character-level trigrams** from vowel sequences
- Rank trigrams by frequency (top result: `eie` with 76 occurrences)

---

## Skills Demonstrated

### Technical Skills
- **Python** — data structures, loops, string manipulation, list comprehensions
- **NLTK** — tokenization (`word_tokenize`), stopword filtering, lemmatization (`WordNetLemmatizer`), frequency distributions (`FreqDist`), n-gram extraction (`ngrams`), corpus access (Reuters, Movie Reviews, State of the Union, UDHR)
- **NumPy** — building and populating a co-occurrence matrix
- **Matplotlib** — bar charts, word cloud rendering, figure formatting
- **WordCloud** — generating visual summaries of text data

### NLP Concepts
- Text preprocessing pipeline (lowercase → tokenize → filter → normalize)
- Stopword removal
- Lemmatization vs. stemming
- N-gram language modeling (bigrams, trigrams)
- Co-occurrence matrices as the basis for word embeddings
- Corpus linguistics and working with labeled datasets

---

## Why This Project is Worth Showcasing

This lab demonstrates the **complete foundational NLP pipeline** that underpins real-world applications like search engines, sentiment analysis tools, recommendation systems, and large language models. The progression from raw text → cleaning → analysis → visualization mirrors how NLP pipelines are built in industry and research.

Key takeaways relevant to roles in **Data Science**, **Machine Learning**, and **Software Engineering**:
- Hands-on experience with the industry-standard NLTK toolkit
- Understanding of how text is transformed into structured numerical representations (co-occurrence matrices are the precursor to word2vec and transformer embeddings)
- Ability to apply statistical methods (frequency distributions, n-grams) to draw insights from unstructured text data

---

## Tech Stack

- Python 3
- NLTK
- NumPy
- Matplotlib
- WordCloud
- Jupyter Notebook
