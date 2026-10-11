"""Resume recommendation scoring."""
from __future__ import annotations
import numpy as np
from sklearn.neighbors import NearestNeighbors

# The paper's KNN flow uses cosine *distance* to identify nearest resumes and
# filters at 0.7. Cosine similarity is displayed to the user as a percentage.
RELEVANCE_DISTANCE_THRESHOLD = 0.7
RECOMMENDATION_THRESHOLD = 0.70
# All matching and project weights live here for straightforward tuning.
OVERALL_SCORE_WEIGHTS = {"skill": 0.50, "project": 0.30, "semantic": 0.20}
PROJECT_RELEVANCE_WEIGHTS = {"skill": 0.40, "technology": 0.25, "domain": 0.20, "functional": 0.15}

def is_relevant_similarity(similarity: float) -> bool:
    """Return True when the paper threshold is met in cosine-distance space."""
    return (1 - similarity) <= RELEVANCE_DISTANCE_THRESHOLD


def combined_match_score(skill_score: float | None, project_score: float | None,
                         semantic_score: float) -> float:
    """Combine skills, project evidence, and whole-document semantic similarity.

    Missing components are excluded and the remaining configured weights are
    renormalised.  Therefore resumes without a Projects section still receive
    a fair skill/semantic score instead of an automatic 30% penalty.
    """
    components = {"semantic": semantic_score}
    if skill_score is not None:
        components["skill"] = skill_score
    if project_score is not None:
        components["project"] = project_score
    available_weight = sum(OVERALL_SCORE_WEIGHTS[name] for name in components)
    return sum(OVERALL_SCORE_WEIGHTS[name] * score for name, score in components.items()) / available_weight


def is_recommended(combined_score: float) -> bool:
    return combined_score >= RECOMMENDATION_THRESHOLD

def validate_job_description(cleaned_text: str) -> None:
    if len(cleaned_text.split()) < 5:
        raise ValueError("Please provide at least five meaningful words in the job description.")

def rank_resumes(job_features, resume_features, records: list[dict], top_k: int = 5,
                 distance_threshold: float = RELEVANCE_DISTANCE_THRESHOLD) -> list[dict]:
    """Fit cosine KNN; retain nearest resumes within the paper's distance threshold."""
    if len(records) == 0:
        return []
    neighbours = NearestNeighbors(n_neighbors=min(top_k, len(records)), metric="cosine")
    neighbours.fit(resume_features)
    distances, indices = neighbours.kneighbors(job_features)
    ranked = []
    for distance, index in zip(distances[0], indices[0]):
        similarity = float(1 - distance)
        if distance <= distance_threshold:
            ranked.append({**records[int(index)], "tfidf_similarity": similarity})
    return ranked
