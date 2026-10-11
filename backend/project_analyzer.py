"""Extract resume projects and score how well they demonstrate JD requirements."""
from __future__ import annotations

import re

from backend.ranking import PROJECT_RELEVANCE_WEIGHTS
from backend.semantic_matcher import semantic_similarity
from backend.skill_matcher import extract_skills, skill_match

PROJECT_HEADING = re.compile(
    r"^(?:(?:academic|personal|major|mini|key)\s+)?(?:projects?|project experience|internships?\s*/?\s*projects?)\s*$",
    re.IGNORECASE,
)
SECTION_HEADING = re.compile(
    r"^(?:education|technical skills?|skills?|experience|work experience|employment|internships?|"
    r"certifications?|achievements?|courses?|training|languages?|interests?|soft skills?|references?)\s*$",
    re.IGNORECASE,
)
ACTION_WORDS = {
    "analyze", "build", "create", "deploy", "design", "develop", "implement", "integrate",
    "maintain", "test", "train", "optimize", "automate", "model", "evaluate",
}


def extract_projects(resume_text: str) -> list[dict]:
    """Extract titled project blocks under common project-section headings.

    The parser intentionally preserves the original project text.  A resume
    without a recognisable project section simply returns an empty list.
    """
    # pypdf may surface a font bullet as DEL (\x7f), not a normal bullet.
    lines = [line.strip(" •\t\x7f\u2022") for line in resume_text.splitlines() if line.strip()]
    start = next((i + 1 for i, line in enumerate(lines) if PROJECT_HEADING.match(line)), None)
    if start is None:
        return []
    section: list[str] = []
    for line in lines[start:]:
        if section and PROJECT_HEADING.match(line):
            break
        if SECTION_HEADING.match(line):
            break
        section.append(line)

    projects: list[dict] = []
    current: dict | None = None
    for line in section:
        # Typical PDF extraction: "Project title: Developed …" or title — description.
        split = re.match(r"^(.{2,90}?)(?:\s*[:\-–—]\s+)(.+)$", line)
        if split and not split.group(1).lower().startswith(("technologies", "skills used")):
            if current:
                projects.append(current)
            current = {"title": split.group(1).strip(), "description": split.group(2).strip()}
        elif current:
            current["description"] += " " + line
    if current:
        projects.append(current)

    for project in projects:
        project["text"] = f'{project["title"]}. {project["description"]}'
        project["skills"] = extract_skills(project["text"])
    return projects


def _functional_relevance(project_text: str, job_description: str) -> float:
    project_actions = {word for word in re.findall(r"[a-z]+", project_text.lower()) if word in ACTION_WORDS}
    job_actions = {word for word in re.findall(r"[a-z]+", job_description.lower()) if word in ACTION_WORDS}
    if not job_actions:
        # A technical project that documents several actions is still stronger
        # evidence than a title-only project when the JD has no action verbs.
        return min(len(project_actions) / 3, 1.0)
    return len(project_actions & job_actions) / len(job_actions)


def analyze_projects(resume_text: str, job_description: str, vectorizer) -> tuple[list[dict], str]:
    """Return explainable per-project relevance analyses and semantic method."""
    projects = extract_projects(resume_text)
    analyses: list[dict] = []
    semantic_method = "TF-IDF cosine fallback"
    for project in projects:
        project_skills = skill_match(job_description, project["text"])
        skill_score = project_skills["score"] if project_skills["score"] is not None else 0.0
        required_count = len(project_skills["required"])
        technology_score = len(project_skills["matched"]) / max(len(project["skills"]), required_count, 1)
        domain_score, semantic_method = semantic_similarity(project["text"], job_description, vectorizer)
        functional_score = _functional_relevance(project["text"], job_description)
        score = (
            PROJECT_RELEVANCE_WEIGHTS["skill"] * skill_score
            + PROJECT_RELEVANCE_WEIGHTS["technology"] * technology_score
            + PROJECT_RELEVANCE_WEIGHTS["domain"] * domain_score
            + PROJECT_RELEVANCE_WEIGHTS["functional"] * functional_score
        )
        if project_skills["matched"]:
            reason = "Demonstrates " + ", ".join(project_skills["matched"]) + "."
        elif project_skills["partial"]:
            related = ", ".join(item["resume_skill"] for item in project_skills["partial"])
            reason = f"Shows related experience in {related}, but not the required technologies."
        else:
            reason = "Limited evidence of the job's required skills or responsibilities."
        analyses.append({
            "title": project["title"], "description": project["description"], "skills": project["skills"],
            "relevant_skills": project_skills["matched"], "partial_skills": project_skills["partial"],
            "missing_skills": project_skills["missing"], "relevance_score": float(score), "reason": reason,
            "components": {"skill_match": float(skill_score), "technology_match": float(technology_score),
                           "domain_relevance": float(domain_score), "functional_relevance": float(functional_score)},
        })
    return sorted(analyses, key=lambda item: item["relevance_score"], reverse=True), semantic_method


def aggregate_project_score(projects: list[dict]) -> float | None:
    """Let the strongest project carry more evidence than secondary projects."""
    if not projects:
        return None
    if len(projects) == 1:
        return projects[0]["relevance_score"]
    return 0.70 * projects[0]["relevance_score"] + 0.30 * projects[1]["relevance_score"]
