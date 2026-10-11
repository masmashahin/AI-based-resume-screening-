"""[BASE PAPER] Text cleaning used before TF-IDF feature extraction."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

PROJECT_NLTK_DATA = Path(__file__).resolve().parents[1] / "data" / "nltk_data"


@lru_cache(maxsize=1)
def _nlp_resources():
    """Load NLTK lazily so the web app can give a useful setup error."""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        # Keep resources project-local when provided by the setup process.
        if str(PROJECT_NLTK_DATA) not in nltk.data.path:
            nltk.data.path.insert(0, str(PROJECT_NLTK_DATA))
        nltk.data.find("corpora/stopwords")
        nltk.data.find("corpora/wordnet")
        return set(stopwords.words("english")), WordNetLemmatizer()
    except (ImportError, LookupError) as error:
        raise RuntimeError(
            "NLTK resources are unavailable. Run: python -m nltk.downloader "
            "stopwords wordnet omw-1.4"
        ) from error


def preprocess_text(text: object) -> str:
    """Remove non-letters, lowercase, tokenize, remove stop words, lemmatize.

    This deliberately mirrors Section 3.2.2 of the reference paper. No stemming,
    named-entity removal, or other extra transformation is applied.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    letters_only = re.sub(r"[^a-zA-Z]+", " ", text).lower()
    tokens = re.findall(r"[a-z]+", letters_only)
    stop_words, lemmatizer = _nlp_resources()
    return " ".join(lemmatizer.lemmatize(token) for token in tokens if token not in stop_words)
