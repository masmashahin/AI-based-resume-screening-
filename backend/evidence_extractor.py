"""Section-aware extraction of skill evidence from resume text."""
from __future__ import annotations

import re

from backend.project_analyzer import extract_projects
from backend.skill_matcher import SKILL_PATTERNS, extract_skills

SECTION_PATTERNS = {
    "skills": r"^(?:technical )?skills?$", "summary": r"^(?:professional )?(?:summary|profile|objective)$",
    "experience": r"^(?:work )?experience$", "internship": r"^internships?$",
    "certification": r"^certifications?$", "education": r"^education$",
}


def _has_affirmative_mention(skill: str, text: str) -> bool:
    """Do not treat statements such as 'no demonstrated BERT' as evidence."""
    lower = text.lower()
    for pattern in SKILL_PATTERNS[skill]:
        for match in re.finditer(pattern, lower):
            sentence_start = max(lower.rfind(".", 0, match.start()), lower.rfind("\n", 0, match.start())) + 1
            lead = lower[sentence_start:match.start()]
            if not re.search(r"\b(?:no|without|lacks?|not)\b", lead):
                return True
    return False


def extract_resume_evidence(resume_text: str) -> dict:
    """Return section/project evidence without counting repeated mentions as new skills."""
    lines = [line.strip(" •\t\x7f\u2022") for line in resume_text.splitlines() if line.strip()]
    sections = {name: [] for name in SECTION_PATTERNS}
    current = "summary"
    for line in lines:
        matched = next((name for name, pattern in SECTION_PATTERNS.items() if re.match(pattern, line, re.I)), None)
        if matched:
            current = matched
        else:
            sections.setdefault(current, []).append(line)
    projects = extract_projects(resume_text)
    entries = []
    for source, content in sections.items():
        text = " ".join(content)
        for skill in extract_skills(text):
            if not _has_affirmative_mention(skill, text):
                continue
            entries.append({"skill": skill, "source": source, "context": text[:300], "project_name": None})
    for project in projects:
        for skill in project["skills"]:
            entries.append({"skill": skill, "source": "project", "context": project["description"],
                            "project_name": project["title"]})
    # Evidence instances are intentionally retained; aggregation happens later
    # with diminishing returns per source type.
    return {"sections": sections, "projects": projects, "entries": entries,
            "candidate_skills": sorted({entry["skill"] for entry in entries})}
