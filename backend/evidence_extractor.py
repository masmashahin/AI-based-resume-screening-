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


def _is_non_evidence_context(skill: str, sentence: str, start: int, end: int) -> bool:
    """Return whether this *individual* skill mention is not candidate evidence."""
    before, after = sentence[:start], sentence[end:]
    # A contrast after a negative statement can still contain a positive skill:
    # "No Python, but experienced in SQL" must retain SQL.
    local_before = re.split(r"\b(?:but|however|whereas)\b|;", before)[-1]
    negative_before = r"\b(?:no|without|lack(?:s|ing)?|never)\b[^.\n]{0,160}$"
    negated_before = r"\b(?:do|does|did|have|has|had|is|are|was|were)\s+not\b[^.\n]{0,160}$"
    if re.search(negative_before, local_before) or re.search(negated_before, local_before):
        return True
    if re.search(r"\b(?:rather than|instead of|unlike)\s*$", local_before):
        return True
    if re.search(r"^\s+(?:is|was|are|were)\s+not\b[^.\n]{0,45}", after):
        return True
    if re.search(r"^\s+(?:is|was|are|were)\s+(?:required|preferred|mandatory|essential)\b", after):
        return True
    if re.search(r"\b(?:looking|want(?:s)?|hope|plan(?:s|ning)?|aspir(?:e|ing)|interested)\b[^.\n]{0,35}\blearn(?:ing)?\b[^.\n]{0,25}$", local_before):
        return True
    if re.search(r"\b(?:job|role|position|requirement)s?\s*[:\-][^.\n]*$", before):
        return True
    return False


def _has_affirmative_mention(skill: str, text: str) -> bool:
    """Keep demonstrated mentions while excluding negative or requirement-only ones."""
    lower = text.lower()
    for pattern in SKILL_PATTERNS[skill]:
        for match in re.finditer(pattern, lower):
            sentence_start = max(lower.rfind(".", 0, match.start()), lower.rfind("!", 0, match.start()),
                                 lower.rfind("?", 0, match.start()), lower.rfind("\n", 0, match.start())) + 1
            sentence_end = min((boundary for boundary in (lower.find(".", match.end()), lower.find("!", match.end()),
                                                           lower.find("?", match.end()), lower.find("\n", match.end()))
                                if boundary >= 0), default=len(lower))
            sentence = lower[sentence_start:sentence_end]
            start, end = match.start() - sentence_start, match.end() - sentence_start
            if not _is_non_evidence_context(skill, sentence, start, end):
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
        # Project text is also consumed by the project scorer, so remove the
        # same non-evidence mentions before any score can use them.
        project["skills"] = [skill for skill in project["skills"]
                             if _has_affirmative_mention(skill, project["text"])]
        for skill in project["skills"]:
            entries.append({"skill": skill, "source": "project", "context": project["description"],
                            "project_name": project["title"]})
    # Evidence instances are intentionally retained; aggregation happens later
    # with diminishing returns per source type.
    return {"sections": sections, "projects": projects, "entries": entries,
            "candidate_skills": sorted({entry["skill"] for entry in entries})}
