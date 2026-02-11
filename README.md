# Automatic Extractive Notes Summarization using NLP Techniques

**MSc Computer Science Project**

An unsupervised, statistical NLP approach to extractive text summarization using TF-IDF scoring.

---

## Project Overview

This is a Python web application that generates concise summaries from long academic notes. It uses **extractive summarization** — selecting the most important sentences from the original text rather than generating new ones.

| Aspect           | Detail                                  |
|------------------|-----------------------------------------|
| Approach         | Extractive (sentence selection)         |
| Technique        | TF-IDF scoring + category-balanced selection |
| Learning type    | Unsupervised (no training data needed)  |
| External AI APIs | None — fully self-contained             |
| Language         | Python 3                                |

---

## Algorithm

1. **Sentence Tokenization** — Split input into sentences using NLTK's `sent_tokenize`.
2. **Preprocessing** — Clean each sentence (lowercase, remove punctuation, remove stopwords) for scoring only. Original sentences are preserved for output.
3. **TF-IDF Vectorization** — Convert cleaned sentences to numerical vectors. Words that are frequent in one sentence but rare overall get the highest weight.
4. **Sentence Scoring** — Sum all TF-IDF values per sentence to get an importance score.
5. **Category-Balanced Selection** — Classify sentences into academic categories (causes, impacts, solutions, etc.) and ensure each category is represented before filling remaining slots by rank.
6. **Reordering** — Sort selected sentences back into original order for natural reading flow.

---

## Project Structure

```
notes_summarizer/
├── app.py                  # Flask web server & routes
├── summarizer.py           # Core TF-IDF summarization algorithm
├── requirements.txt        # Python dependencies
├── README.md
├── utils/
│   ├── __init__.py
│   ├── preprocess.py       # Text cleaning pipeline
│   └── pdf_reader.py       # PDF text extraction
├── templates/
│   ├── base.html           # Shared layout template
│   ├── home.html           # Landing page
│   ├── index.html          # Summarizer tool page
│   ├── how_it_works.html   # Algorithm explanation
│   └── about.html          # About the project
└── static/
    └── style.css           # Stylesheet
```

---

## How to Run

### Prerequisites

- Python 3.8+
- pip

### Setup

```bash
cd notes_summarizer

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

On first run, NLTK will download `punkt` and `stopwords` data automatically.

---

## Usage

1. Open the app at `http://127.0.0.1:5000`
2. Navigate to the **Summarizer** page
3. Paste text or upload a PDF
4. Choose summary length: Short (20%), Medium (35%), or Long (50%)
5. Click **Generate Summary**
6. View the summary, statistics, and sentence scores

---

## Technologies

| Technology   | Role                           |
|-------------|--------------------------------|
| Python 3    | Core language                  |
| Flask       | Web framework                  |
| NLTK        | Sentence tokenization & stopwords |
| scikit-learn | TF-IDF vectorization          |
| NumPy       | Numerical operations           |
| pdfplumber  | PDF text extraction            |
| HTML + CSS  | Frontend interface             |

---

## Limitations

- **Extractive only** — cannot generate new sentences or paraphrase.
- **No semantic understanding** — TF-IDF is statistical, not meaning-aware.
- **No cross-sentence reasoning** — each sentence is scored independently.
- **English only** — stopwords and tokenizer are configured for English.
- **PDF layout issues** — complex layouts (multi-column, tables) may produce messy text.

---

## References

- Luhn, H. P. (1958). *The Automatic Creation of Literature Abstracts.* IBM Journal of Research and Development.
- Salton, G. & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval.* Information Processing & Management.
- NLTK — https://www.nltk.org/
- scikit-learn TfidfVectorizer — https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- Flask — https://flask.palletsprojects.com/
# NLP_Project
