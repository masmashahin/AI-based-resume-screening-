"""Structured, importance-aware analysis of a job description."""
from __future__ import annotations

import re

from backend.skill_matcher import SKILL_METADATA, extract_skills

ROLE_TAXONOMY = {
    "NLP / Machine Learning Engineer": {"Natural Language Processing", "Machine Learning", "TensorFlow", "PyTorch", "Python", "Sentence Transformers"},
    "Computer Vision Engineer": {"OpenCV", "TensorFlow", "PyTorch", "Python"},
    "Machine Learning Engineer": {"Machine Learning", "TensorFlow", "PyTorch", "Python"},
    "Data Scientist": {"Machine Learning", "Pandas", "NumPy", "SQL", "Python"},
    "Backend Developer": {"Flask", "Django", "FastAPI", "REST APIs", "PostgreSQL", "MySQL"},
    "Frontend Developer": {"React", "Angular", "JavaScript", "TypeScript", "HTML", "CSS"},
    "Full Stack Developer": {"React", "Flask", "Django", "Node.js", "REST APIs"},
    "DevOps Engineer": {"Docker", "AWS", "Git"},
}


def _importance(text: str, skill: str, position: int, total: int) -> tuple[float, bool]:
    lower = text.lower()
    aliases = SKILL_METADATA[skill]["aliases"]
    location = min((lower.find(alias) for alias in aliases if lower.find(alias) >= 0), default=len(lower))
    window = lower[max(0, location - 70):location + 100]
    # Explicit requirement lists often contain several comma-separated skills;
    # inspect the current list label, rather than only a tiny local window.
    preceding = lower[max(0, location - 350):location]
    labels = list(re.finditer(r"\b(required|mandatory|essential|must have|preferred|nice to have|bonus|plus)\b", preceding))
    list_label = labels[-1].group(1) if labels else ""
    if list_label in {"required", "mandatory", "essential", "must have"}:
        return 1.0, True
    if list_label in {"preferred", "nice to have", "bonus", "plus"}:
        return 0.35, False
    if re.search(r"\b(must have|required|mandatory|essential)\b", window):
        return 1.0, True
    if re.search(r"\b(should have|strong|proficient|experience with)\b", window):
        return 0.85, position < 3
    if re.search(r"\b(preferred|nice to have|bonus|plus)\b", window):
        return 0.50, False
    frequency = sum(lower.count(alias) for alias in aliases)
    return min(0.90, 0.70 + (0.10 if position < 3 else 0) + (0.05 if frequency > 1 else 0)), position < max(2, total // 3)


def analyze_job_description(job_description: str) -> dict:
    skills = extract_skills(job_description)
    requirements = []
    for position, skill in enumerate(skills):
        importance, critical = _importance(job_description, skill, position, len(skills))
        requirements.append({"skill": skill, "category": SKILL_METADATA[skill]["category"],
                             "importance": importance, "critical": critical})
    role, confidence, evidence = infer_role(requirements, job_description)
    return {"requirements": requirements, "role": role, "role_confidence": confidence,
            "role_evidence": evidence, "responsibilities": _responsibilities(job_description)}


def _responsibilities(text: str) -> list[str]:
    actions = ("build", "develop", "design", "deploy", "train", "evaluate", "analyze", "implement", "maintain")
    return [action for action in actions if re.search(rf"\b{action}\w*\b", text.lower())]


def infer_role(requirements: list[dict], job_description: str) -> tuple[str, float, list[str]]:
    required = {item["skill"] for item in requirements}
    lower = job_description.lower()
    scored = []
    for role, signals in ROLE_TAXONOMY.items():
        skill_score = len(required & signals) / max(len(signals), 1)
        title_bonus = 0.20 if any(word in lower for word in role.lower().replace("/", " ").split()) else 0.0
        scored.append((min(1.0, skill_score + title_bonus), role, sorted(required & signals)))
    score, role, evidence = max(scored, default=(0.0, "Role not identified", []))
    return role, score, evidence
