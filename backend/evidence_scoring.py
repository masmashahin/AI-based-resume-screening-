"""Evidence-weighted matching and deterministic candidate scoring."""
from __future__ import annotations

import re

from backend.scoring_config import SCORING_CONFIG
from backend.semantic_matcher import semantic_similarity
from backend.skill_matcher import CAPABILITY_GROUPS

DEPTH_SIGNALS = {"trained", "evaluated", "deployment", "deployed", "api", "database", "pipeline", "model",
                 "embedding", "transformer", "classifier", "testing", "docker", "cloud", "optimization"}
ACTION_SIGNALS = {"build", "develop", "design", "deploy", "train", "evaluate", "analyze", "implement", "test", "integrate"}
ROLE_FUNCTION_SIGNALS = {"semantic", "similarity", "search", "retrieval", "embedding", "transformer",
                         "classification", "classifier", "nlp", "language", "document", "text"}


def capability_similarity(required: str, candidate: str) -> tuple[float, str]:
    config = SCORING_CONFIG["capability_similarity"]
    if required == candidate:
        return config["exact"], "Exact"
    for family, members in CAPABILITY_GROUPS.items():
        if required in members and candidate in members:
            relation = "Strongly related" if family == "source_control" else "Related capability"
            score = config["strong_related"] if family == "source_control" else config["family_related"]
            return score, relation
    return 0.0, "Missing"


def _diminishing_strength(sources: set[str]) -> float:
    values = sorted((SCORING_CONFIG["source_strength"].get(source, 0.45) for source in sources), reverse=True)
    if not values:
        return 0.0
    return min(1.0, values[0] + 0.15 * sum(values[1:]))


def score_requirements(requirements: list[dict], evidence: dict) -> tuple[list[dict], float, float, float]:
    """Score weighted requirements and aggregate repeated evidence safely."""
    entries = evidence["entries"]
    analyses = []
    for requirement in requirements:
        alternatives = []
        for entry in entries:
            similarity, relation = capability_similarity(requirement["skill"], entry["skill"])
            if similarity:
                alternatives.append((similarity, relation, entry))
        best_similarity = max((item[0] for item in alternatives), default=0.0)
        selected = [item for item in alternatives if item[0] == best_similarity]
        sources = {item[2]["source"] for item in selected}
        evidence_strength = _diminishing_strength(sources)
        related = sorted({item[2]["skill"] for item in selected if item[2]["skill"] != requirement["skill"]})
        projects = sorted({item[2]["project_name"] for item in selected if item[2]["project_name"]})
        analyses.append({**requirement, "match_strength": best_similarity, "relation": selected[0][1] if selected else "Missing",
                         "evidence_strength": evidence_strength, "sources": sorted(sources),
                         "related_skills": related, "projects": projects,
                         "contribution": requirement["importance"] * best_similarity * evidence_strength})
    total = sum(item["importance"] for item in analyses) or 1.0
    skill_score = sum(item["contribution"] for item in analyses) / total
    evidence_score = sum(item["importance"] * item["evidence_strength"] for item in analyses) / total
    critical = [item for item in analyses if item["critical"]]
    critical_score = sum(item["importance"] * item["match_strength"] * item["evidence_strength"] for item in critical) / (sum(item["importance"] for item in critical) or 1.0)
    return analyses, skill_score, evidence_score, critical_score


def _project_score(project: dict, requirements: list[dict], vectorizer, job_description: str) -> tuple[dict, str]:
    project_evidence = {"entries": [{"skill": skill, "source": "project", "context": project["description"],
                                      "project_name": project["title"]} for skill in project["skills"]], "projects": [project]}
    requirement_analysis, skill_score, _, _ = score_requirements(requirements, project_evidence)
    direct = [item for item in requirement_analysis if item["match_strength"] >= 1.0]
    related = [item for item in requirement_analysis if 0 < item["match_strength"] < 1.0]
    # A project is judged primarily against required capabilities. Preferred
    # tooling can strengthen a match but cannot make its absence look like a
    # failure of an otherwise directly relevant project.
    required = [item for item in requirement_analysis if item["critical"]]
    preferred = [item for item in requirement_analysis if not item["critical"]]
    def coverage(items: list[dict]) -> float:
        return sum(item["importance"] * item["match_strength"] for item in items) / (sum(item["importance"] for item in items) or 1)
    required_coverage = coverage(required) if required else coverage(requirement_analysis)
    preferred_coverage = coverage(preferred) if preferred else 0.0
    # Direct project evidence deserves a high score; do not multiply it by the
    # section-strength again after score_requirements already accounted for it.
    skill_evidence = required_coverage
    technology = min(1.0, 0.85 * required_coverage + 0.15 * preferred_coverage)
    semantic, method = semantic_similarity(project["text"], job_description, vectorizer)
    words = set(re.findall(r"[a-z]+", project["text"].lower()))
    job_words = set(re.findall(r"[a-z]+", job_description.lower()))
    functional = len((words & ACTION_SIGNALS) & (job_words & ACTION_SIGNALS)) / max(len(job_words & ACTION_SIGNALS), 1)
    # Resume project descriptions are frequently noun phrases ("Semantic
    # Search API") rather than complete action sentences. Recognise their
    # domain/function terminology without treating it as a keyword count.
    shared_function_signals = (words & job_words & ROLE_FUNCTION_SIGNALS)
    functional = max(functional, min(1.0, 0.30 * len(shared_function_signals)))
    depth = min(1.0, 0.12 * len(project["skills"]) + 0.10 * len(words & DEPTH_SIGNALS))
    # TF-IDF can understate short, dense project descriptions. Direct
    # requirement evidence is therefore allowed to substantiate the domain.
    domain = min(1.0, max(0.45 * semantic + 0.55 * required_coverage,
                          0.65 * required_coverage + 0.35 * functional))
    weights = SCORING_CONFIG["project_weights"]
    score = (weights["skill_evidence"] * skill_evidence + weights["technology"] * technology + weights["domain"] * domain
             + weights["functional"] * functional + weights["technical_depth"] * depth + weights["semantic"] * semantic)
    reason = ("Directly demonstrates " + ", ".join(item["skill"] for item in direct) + ".") if direct else (
        "Shows related capability in " + ", ".join(item["skill"] for item in related) + "." if related else "No direct evidence of the role requirements.")
    return {"title": project["title"], "description": project["description"], "skills": project["skills"],
            "relevant_skills": [item["skill"] for item in direct], "related_skills": [item["skill"] for item in related],
            "relevance_score": score, "reason": reason,
            "components": {"skill_evidence": skill_evidence, "technology": technology, "domain": domain,
                           "functional": functional, "technical_depth": depth, "semantic": semantic,
                           "required_coverage": required_coverage, "preferred_coverage": preferred_coverage,
                           "shared_function_signals": sorted(shared_function_signals),
                           "raw_project_score": score},
            "requirement_evidence": requirement_analysis}, method


def score_projects(projects: list[dict], requirements: list[dict], vectorizer, job_description: str) -> tuple[list[dict], float | None, str]:
    scored, method = [], "TF-IDF cosine fallback"
    for project in projects:
        analysis, method = _project_score(project, requirements, vectorizer, job_description)
        scored.append(analysis)
    scored.sort(key=lambda item: (-item["relevance_score"], item["title"].lower()))
    if not scored:
        return scored, None, method
    aggregate = scored[0]["relevance_score"]
    if len(scored) > 1:
        aggregate = (SCORING_CONFIG["project_aggregation"]["strongest"] * scored[0]["relevance_score"]
                     + SCORING_CONFIG["project_aggregation"]["second"] * scored[1]["relevance_score"])
    return scored, aggregate, method


def recommendation_label(score: float) -> str:
    return next(label for threshold, label in SCORING_CONFIG["thresholds"] if score >= threshold)


def score_candidate(jd: dict, evidence: dict, resume_text: str, vectorizer) -> dict:
    reqs, skill_score, evidence_score, critical = score_requirements(jd["requirements"], evidence)
    projects, project_score, project_method = score_projects(evidence["projects"], jd["requirements"], vectorizer, jd["raw_text"])
    document_semantic, semantic_method = semantic_similarity(resume_text, jd["raw_text"], vectorizer)
    # Requirement-level semantic evidence complements whole-document similarity:
    # dense resumes should not lose credit merely because unrelated sections
    # dilute a single JD-to-resume embedding/cosine comparison.
    importance_total = sum(item["importance"] for item in reqs) or 1.0
    requirement_semantic = sum(
        item["importance"] * min(1.0, 0.80 * item["match_strength"] + 0.20 * item["evidence_strength"])
        for item in reqs
    ) / importance_total
    semantic = 0.75 * requirement_semantic + 0.25 * document_semantic
    components = {"skill": skill_score, "semantic": semantic, "evidence": evidence_score}
    if project_score is not None:
        components["project"] = project_score
    final_weights = SCORING_CONFIG["final_weights"]
    total_weight = sum(final_weights[name] for name in components)
    base_score = sum(final_weights[name] * value for name, value in components.items()) / total_weight
    gate = SCORING_CONFIG["critical_gate"]
    final_score = base_score * (gate["floor"] + gate["coverage_weight"] * critical)
    matched = [item["skill"] for item in reqs if item["match_strength"] >= 1.0]
    related = [{"required_skill": item["skill"], "resume_skills": item["related_skills"], "relation": item["relation"]}
               for item in reqs if 0 < item["match_strength"] < 1.0]
    missing = [item["skill"] for item in reqs if not item["match_strength"]]
    return {"overall_score": final_score, "skill_score": skill_score, "project_score": project_score,
            "semantic_similarity": semantic, "evidence_strength": evidence_score, "critical_coverage": critical,
            "recommendation": recommendation_label(final_score), "requirements": reqs, "matched_skills": matched,
            "related_skills": related, "missing_skills": missing, "projects": projects,
            "semantic_method": semantic_method if semantic_method != "TF-IDF cosine fallback" else project_method,
            "debug": {"components": components, "base_score": base_score,
                      "semantic_breakdown": {"document": document_semantic, "requirement_level": requirement_semantic},
                      "critical_gate_factor": gate["floor"] + gate["coverage_weight"] * critical}}
