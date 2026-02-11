"""
summarizer.py - Core extractive summarization engine.

Implements TF-IDF based sentence scoring with category-aware
balanced selection. This is extractive: we select existing sentences,
we do not generate new text. No neural networks or external APIs.

Algorithm overview:
    1. Tokenize input into sentences (NLTK)
    2. Clean sentences for scoring (lowercase, remove punctuation/stopwords)
    3. Vectorize with TF-IDF (scikit-learn)
    4. Score each sentence by summing its TF-IDF values
    5. Classify sentences into academic categories via keyword matching
    6. Select sentences: guarantee one per category, fill rest by rank
    7. Reorder selected sentences to match original text order
"""

import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.preprocess import clean_text

# Download tokenizer data (runs once, silent if already present)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# Fraction of sentences to keep for each summary length
SUMMARY_RATIOS = {
    "short":  0.20,
    "medium": 0.35,
    "long":   0.50,
}

# Category keywords for academic content classification.
# If a sentence contains any keyword from a category, it is tagged
# with that category. A sentence can belong to multiple categories.
CATEGORY_KEYWORDS = {
    "Causes": [
        "cause", "caused", "causes", "because", "due to", "result of",
        "reason", "factor", "contribute", "contributing", "led to",
        "leads to", "leading to", "driven by", "origin", "source",
        "responsible", "emission", "emissions", "greenhouse", "carbon",
        "fossil fuel", "deforestation", "pollution", "burning",
        "industrial", "human activity", "anthropogenic",
    ],
    "Environmental Impact": [
        "sea level", "glacier", "glaciers", "ice cap", "ice caps",
        "melting", "ocean", "temperature", "warming", "climate",
        "ecosystem", "ecosystems", "biodiversity", "species",
        "extinction", "habitat", "coral", "reef", "weather",
        "precipitation", "atmosphere", "ozone", "arctic", "antarctic",
        "permafrost", "acidification", "desertification", "erosion",
        "wildfire", "hurricane", "cyclone", "typhoon", "storm",
        "environment", "environmental", "ecological", "natural",
    ],
    "Human & Social Impact": [
        "food", "agriculture", "crop", "crops", "harvest", "famine",
        "health", "disease", "mortality", "death", "displacement",
        "migration", "refugee", "poverty", "economic", "economy",
        "livelihood", "community", "communities", "population",
        "society", "social", "infrastructure", "housing", "water supply",
        "drinking water", "sanitation", "malnutrition", "conflict",
        "inequality", "vulnerable", "developing countries",
    ],
    "Consequences": [
        "flood", "floods", "flooding", "drought", "droughts",
        "scarcity", "shortage", "damage", "destruction", "loss",
        "devastation", "crisis", "catastrophe", "disaster", "risk",
        "threat", "impact", "consequence", "effect", "affects",
        "affected", "severe", "extreme", "intensify", "worsen",
    ],
    "Solutions": [
        "solution", "policy", "policies", "mitigation", "adaptation",
        "reduce", "renewable", "solar", "wind", "energy", "sustainable",
        "sustainability", "legislation", "regulation", "agreement",
        "treaty", "paris agreement", "carbon neutral", "net zero",
        "reforestation", "recycle", "recycling", "innovation",
        "technology", "collaborate", "collaboration", "invest",
        "investment", "strategy", "plan", "action", "initiative",
        "protect", "conservation", "transition", "infrastructure",
        "government", "international", "framework", "awareness",
    ],
}


def _classify_sentence(sentence_lower):
    """Return list of category names that match the given sentence."""
    matched = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in sentence_lower:
                matched.append(category)
                break
    return matched


def summarize(text, length="medium"):
    """Generate a category-balanced extractive summary.

    Args:
        text:   Raw input text to summarize.
        length: "short", "medium", or "long".

    Returns:
        dict with keys: summary, num_original, num_summary,
        sentence_scores, categories_found, categories_covered.
    """

    # Step 1: Sentence tokenization
    original_sentences = sent_tokenize(text)

    if len(original_sentences) <= 2:
        return {
            "summary": text.strip(),
            "num_original": len(original_sentences),
            "num_summary": len(original_sentences),
            "sentence_scores": [(s, 1.0) for s in original_sentences],
            "categories_found": [],
            "categories_covered": [],
        }

    # Step 2: Preprocess for scoring (originals are kept intact)
    cleaned_sentences = [clean_text(s) for s in original_sentences]

    # Filter out sentences that become empty after cleaning
    valid_indices = [i for i, s in enumerate(cleaned_sentences) if s.strip()]
    valid_cleaned = [cleaned_sentences[i] for i in valid_indices]
    valid_originals = [original_sentences[i] for i in valid_indices]

    if len(valid_cleaned) <= 2:
        summary_text = " ".join(valid_originals)
        return {
            "summary": summary_text,
            "num_original": len(original_sentences),
            "num_summary": len(valid_originals),
            "sentence_scores": [(s, 1.0) for s in valid_originals],
            "categories_found": [],
            "categories_covered": [],
        }

    # Step 3: TF-IDF vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(valid_cleaned)

    # Step 4: Score each sentence (sum of TF-IDF values)
    scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    # Step 5: Classify sentences into categories
    sentence_categories = []
    category_to_indices = {}

    for idx, sent in enumerate(valid_originals):
        cats = _classify_sentence(sent.lower())
        sentence_categories.append(cats)
        for cat in cats:
            if cat not in category_to_indices:
                category_to_indices[cat] = []
            category_to_indices[cat].append(idx)

    categories_found = list(category_to_indices.keys())

    # Step 6: Balanced selection
    ratio = SUMMARY_RATIOS.get(length, 0.35)
    num_to_select = max(1, int(len(valid_originals) * ratio))

    selected_set = set()

    # Phase A: one sentence per category (highest score in that category)
    for cat in categories_found:
        best_idx = max(category_to_indices[cat], key=lambda i: scores[i])
        selected_set.add(best_idx)

    # Phase B: fill remaining slots from top-ranked sentences
    ranked_indices = np.argsort(scores)[::-1]
    for idx in ranked_indices:
        if len(selected_set) >= num_to_select:
            break
        selected_set.add(idx)

    # Step 7: Reorder by original position
    selected_indices = sorted(selected_set)
    summary_sentences = [valid_originals[i] for i in selected_indices]
    summary_text = " ".join(summary_sentences)

    # Determine which categories are covered in the final summary
    categories_covered = set()
    for i in selected_indices:
        for cat in sentence_categories[i]:
            categories_covered.add(cat)
    categories_covered = sorted(categories_covered)

    # Build scored list for display (all sentences, sorted by score desc)
    scored_list = sorted(
        zip(valid_originals, scores.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "summary": summary_text,
        "num_original": len(original_sentences),
        "num_summary": len(selected_indices),
        "sentence_scores": scored_list,
        "categories_found": categories_found,
        "categories_covered": categories_covered,
    }
