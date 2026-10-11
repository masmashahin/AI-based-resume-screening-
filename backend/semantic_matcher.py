"""Local semantic similarity with a deterministic offline fallback."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _local_embedding_model():
    """Load an already-downloaded embedding model, never download one at runtime."""
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(MODEL_NAME, local_files_only=True)
    except Exception:
        return None


def semantic_similarity(left: str, right: str, fallback_vectorizer=None) -> tuple[float, str]:
    """Return a 0–1 semantic score and the algorithm actually used.

    The preferred local model is MiniLM.  When it has not been downloaded, the
    existing trained TF-IDF representation keeps the application fully local
    and runnable instead of attempting a network download.
    """
    model = _local_embedding_model()
    if model is not None:
        embeddings = model.encode([left, right], normalize_embeddings=True)
        return float(np.clip(np.dot(embeddings[0], embeddings[1]), 0, 1)), "MiniLM embeddings"
    if fallback_vectorizer is None:
        return 0.0, "unavailable"
    features = fallback_vectorizer.transform([left, right])
    return float((features[0] @ features[1].T).toarray()[0, 0]), "TF-IDF cosine fallback"
