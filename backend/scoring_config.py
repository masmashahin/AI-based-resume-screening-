"""Single source of truth for evidence-weighted screening configuration."""
from __future__ import annotations

SCORING_CONFIG = {
    "final_weights": {"skill": 0.35, "project": 0.35, "semantic": 0.20, "evidence": 0.10},
    "project_weights": {"skill_evidence": 0.30, "technology": 0.20, "domain": 0.15,
                        "functional": 0.15, "technical_depth": 0.10, "semantic": 0.10},
    "source_strength": {"skills": 0.45, "summary": 0.60, "project": 0.90,
                        "experience": 0.78, "internship": 0.78, "certification": 0.65,
                        "education": 0.30},
    "capability_similarity": {"exact": 1.00, "strong_related": 0.82, "family_related": 0.62},
    # Critical coverage is a guardrail. It prevents a high absolute score with
    # missing essentials, but does not repeatedly halve otherwise strong work.
    "critical_gate": {"floor": 0.75, "coverage_weight": 0.25},
    "project_aggregation": {"strongest": 0.85, "second": 0.15},
    "thresholds": ((0.85, "Excellent Match"), (0.75, "Strong Match"), (0.65, "Good Match"),
                   (0.50, "Moderate Match"), (0.35, "Weak Match"), (0.0, "Poor Match")),
}
