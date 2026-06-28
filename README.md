# 🛡️ SMS Spam Detector — NLP Web Application

> **SAIA 2163 — Natural Language Processing | Final Project**  
> Universiti Teknologi Malaysia (UTM) · Faculty of Artificial Intelligence

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8-green)](https://www.nltk.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [NLP Pipeline](#-nlp-pipeline)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Running the App](#-running-the-app)
- [Dataset](#-dataset)
- [Models & Results](#-models--results)
- [Visualizations](#-visualizations)
- [Team Members](#-team-members)

---

## 📌 Project Overview

This project builds a complete **SMS Spam Detection system** using Natural Language Processing (NLP) and Machine Learning. Users can paste any SMS or email message into the web application and receive an instant prediction — **Spam** or **Ham (Legitimate)** — along with a confidence score.

| Item | Detail |
|------|--------|
| **Course** | SAIA 2163 — Natural Language Processing |
| **Theme** | Email / SMS Spam Detector |
| **Dataset** | SMS Spam Collection — 1,000 labeled messages |
| **Models** | Naive Bayes · Logistic Regression |
| **Features** | Bag of Words (BoW) · TF-IDF |
| **App** | Interactive Streamlit web application |
| **Best Accuracy** | 100.00% (on test set) |

---

## ⚙️ NLP Pipeline

```
Raw SMS Text
     │
     ▼
┌─────────────────────────────┐
│  1. Lowercase               │  "FREE PRIZE" → "free prize"
│  2. Remove URLs             │  removes http/www links
│  3. Remove special chars    │  keeps only letters & spaces
│  4. Tokenization            │  "free prize" → ["free", "prize"]
│  5. Stopword Removal        │  removes "the", "is", "and" etc.
│  6. Lemmatization           │  "running" → "run"
└─────────────────────────────┘
     │
     ▼
Feature Extraction (TWO methods)
  ├── Bag of Words (CountVectorizer)
  └── TF-IDF (TfidfVectorizer)
     │
     ▼
Model Training (TWO models)
  ├── Naive Bayes (MultinomialNB)
  └── Logistic Regression
     │
     ▼
Prediction: SPAM 🚫 or HAM ✅
```

---

## ✨ Features

### 5-Page Streamlit Application

| Page | Description |
|------|-------------|
| 🏠 **Home & About** | Project overview, pipeline summary, team info |
| 🔍 **Text Analyzer** | Real-time spam/ham prediction with confidence scores |
| 📊 **Data Explorer** | Browse dataset, statistics, interactive charts |
| 📈 **Visualizations** | Word clouds, confusion matrices, model comparison |
| 🤖 **Model Info** | Model explanations, performance metrics, preprocessing details |

### Key Capabilities
- ✅ Real-time prediction on any text input
- ✅ Confidence score display (spam % vs ham %)
- ✅ Preprocessing breakdown — shows how text is cleaned
- ✅ Spam indicator keyword highlighting
- ✅ Batch analysis — classify multiple messages at once
- ✅ Interactive Plotly charts
- ✅ 6 visualizations included
- ✅ Model selector — switch between 4 model combinations

---

## 📁 Project Structure

```
spam_detector/
│
├── app.py                          # Main Streamlit application (5 pages)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
│
├── data/
│   ├── spam.csv                    # Raw dataset (1,000 messages)
│   ├── spam_preprocessed.csv       # Cleaned dataset with features
│   ├── viz_class_distribution.png  # Visualization: class counts
│   ├── viz_wordclouds.png          # Visualization: word clouds
│   ├── viz_top_words.png           # Visualization: top 20 words
│   ├── viz_model_comparison.png    # Visualization: model performance
│   ├── viz_confusion_matrices.png  # Visualization: confusion matrices
│   └── viz_length_distribution.png # Visualization: message lengths
│
├── models/
│   ├── nb_bow.pkl                  # Naive Bayes + BoW model
│   ├── nb_tfidf.pkl                # Naive Bayes + TF-IDF model
│   ├── lr_bow.pkl                  # Logistic Regression + BoW model
│   ├── lr_tfidf.pkl                # Logistic Regression + TF-IDF model
│   ├── bow_vectorizer.pkl          # Fitted CountVectorizer
│   └── tfidf_vectorizer.pkl        # Fitted TfidfVectorizer
│
└── notebooks/
    └── spam_detector_nlp_pipeline.ipynb   # Full NLP pipeline notebook
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/spam-detector.git
cd spam-detector
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

> **Windows users:** If `pip` is not recognised, use `python -m pip install -r requirements.txt`

### Step 3 — Run the application
```bash
streamlit run app.py
```

> **Windows users:** If `streamlit` is not recognised, use `python -m streamlit run app.py`

The app will open automatically at **http://localhost:8501**

---

## 🖥️ Running the App

Once running, navigate using the **sidebar on the left**:

1. **Home** — Start here for a project overview
2. **Text Analyzer** — Paste a message and click *Analyze*
3. **Data Explorer** — Filter and explore the dataset
4. **Visualizations** — View all charts in tabs
5. **Model Info** — Understand how the models work

### Example predictions

| Message | Prediction |
|---------|-----------|
| `WINNER!! Call 09061701461 to claim your FREE prize NOW!` | 🚫 SPAM |
| `Hey, are you coming to the study group tonight?` | ✅ HAM |
| `URGENT: Your account is suspended. Verify now.` | 🚫 SPAM |
| `Can you pick up some milk on your way home?` | ✅ HAM |

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Name** | SMS Spam Collection |
| **Size** | 1,000 messages |
| **Classes** | Spam (500) · Ham (500) |
| **Balance** | 50% / 50% (perfectly balanced) |
| **Format** | CSV — columns: `label`, `message` |
| **Source** | Custom SMS dataset based on real spam patterns |

**Sample data:**

| Label | Message |
|-------|---------|
| spam | `WINNER!! You have been selected to receive a 900 prize! Call 09061701461 to claim NOW!` |
| ham | `Hey, are you coming to the study group tonight at 7pm?` |
| spam | `URGENT: Your account will be suspended. Verify your details immediately.` |
| ham | `Can you pick up some milk on your way home please?` |

---

## 🤖 Models & Results

### Feature Extraction Methods

| Method | Description | Library |
|--------|-------------|---------|
| **Bag of Words (BoW)** | Word occurrence counts; simple baseline | `CountVectorizer` (sklearn) |
| **TF-IDF** | Term frequency × inverse document frequency; weights distinctive words | `TfidfVectorizer` (sklearn) |

Both methods use:
- `max_features = 3,000`
- `ngram_range = (1, 2)` — unigrams and bigrams

### Classification Models

| Model | Description |
|-------|-------------|
| **Naive Bayes** | Probabilistic classifier based on Bayes' theorem; fast and highly effective for text |
| **Logistic Regression** | Linear classifier with probability output; best confidence calibration |

### Performance Results

| Model + Features | Accuracy | Precision | Recall | F1-Score |
|-----------------|----------|-----------|--------|----------|
| Naive Bayes + BoW | 100.00% | 100.00% | 100.00% | 100.00% |
| Naive Bayes + TF-IDF | 100.00% | 100.00% | 100.00% | 100.00% |
| Logistic Regression + BoW | 100.00% | 100.00% | 100.00% | 100.00% |
| **Logistic Regression + TF-IDF** ✅ | **100.00%** | **100.00%** | **100.00%** | **100.00%** |

> **Best model:** Logistic Regression + TF-IDF — selected for best probability calibration and generalisability.

**Evaluation method:** 80/20 stratified train-test split (`random_state=42`)

---

## 📈 Visualizations

The project includes **6 visualizations**:

| # | Chart | Description |
|---|-------|-------------|
| 1 | **Word Cloud (Spam)** | Most frequent words in spam messages |
| 2 | **Word Cloud (Ham)** | Most frequent words in legitimate messages |
| 3 | **Class Distribution** | Bar + pie chart of spam vs ham counts |
| 4 | **Message Length Distribution** | Histogram and box plot by class |
| 5 | **Top 20 Words** | Most frequent words per class (bar chart) |
| 6 | **Confusion Matrices** | All 4 model combinations side by side |
| 7 | **Model Comparison** | Accuracy/Precision/Recall/F1 grouped bar chart |

---

## 👥 Team Members

| Member | Responsibilities |
|--------|----------------|
| **Member A** | Data collection · Text preprocessing · NLP pipeline · Model training · Technical report |
| **Member B** | Streamlit application · Visualizations · GitHub repository · Deployment |

---

## 📚 Libraries & References

| Library | Version | Purpose |
|---------|---------|---------|
| `streamlit` | 1.35.0 | Web application framework |
| `scikit-learn` | 1.4.2 | ML models and feature extraction |
| `nltk` | 3.8.1 | NLP preprocessing (tokenization, stopwords, lemmatization) |
| `pandas` | 2.2.2 | Data manipulation |
| `numpy` | 1.26.4 | Numerical operations |
| `matplotlib` | 3.9.0 | Static visualizations |
| `seaborn` | 0.13.2 | Statistical visualizations |
| `plotly` | 5.22.0 | Interactive charts |
| `wordcloud` | 1.9.3 | Word cloud generation |
| `joblib` | 1.4.2 | Model serialization |

---

*SAIA 2163 — Natural Language Processing · UTM Faculty of Artificial Intelligence*
