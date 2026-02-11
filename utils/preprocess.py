"""
preprocess.py - Text cleaning pipeline for TF-IDF scoring.

Applies: lowercase -> remove punctuation -> remove stopwords.
Only used for scoring; the final summary keeps original sentences.
"""

import re
import string

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

STOP_WORDS = set(stopwords.words("english"))


def to_lowercase(text):
    """Convert text to lowercase."""
    return text.lower()


def remove_punctuation(text):
    """Strip all punctuation characters."""
    translator = str.maketrans("", "", string.punctuation)
    return text.translate(translator)


def remove_stopwords(text):
    """Remove common English stopwords."""
    words = text.split()
    filtered = [w for w in words if w not in STOP_WORDS]
    return " ".join(filtered)


def remove_extra_whitespace(text):
    """Collapse multiple spaces/newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text):
    """Run the full preprocessing pipeline.

    Order: whitespace -> lowercase -> punctuation -> stopwords.
    """
    text = remove_extra_whitespace(text)
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    return text
