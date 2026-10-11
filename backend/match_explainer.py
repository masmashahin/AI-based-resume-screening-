"""Plain-language explanation of the TF-IDF resume-to-job match."""
from __future__ import annotations

def matching_keywords(vectorizer, resume_features, job_features, limit: int = 6) -> list[str]:
    """Return the highest-contributing shared TF-IDF terms for one match score."""
    shared = resume_features.multiply(job_features).tocoo()
    if shared.nnz == 0:
        return []
    terms = vectorizer.get_feature_names_out()
    strongest = sorted(zip(shared.col, shared.data), key=lambda item: item[1], reverse=True)
    return [str(terms[column]) for column, _ in strongest[:limit]]


def missing_job_keywords(vectorizer, resume_features, job_features, limit: int = 10) -> list[str]:
    """Return high-weight job-description terms that are absent from the resume."""
    job_terms = job_features.tocoo()
    terms = vectorizer.get_feature_names_out()
    missing = [(column, weight) for column, weight in zip(job_terms.col, job_terms.data)
               if resume_features[0, column] == 0]
    strongest = sorted(missing, key=lambda item: item[1], reverse=True)
    return [str(terms[column]) for column, _ in strongest[:limit]]
