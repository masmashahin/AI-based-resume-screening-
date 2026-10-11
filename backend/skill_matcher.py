"""Skill coverage used alongside document-level text similarity."""
from __future__ import annotations

import re

# Canonical skill name -> ways it commonly appears in resumes/job descriptions.
# This small, explicit vocabulary keeps the score explainable to the user.
SKILL_PATTERNS = {
    "Python": (r"\bpython\b",),
    "Java": (r"\bjava\b",),
    "SQL": (r"\bsql\b",),
    "HTML": (r"\bhtml\b",),
    "CSS": (r"\bcss\b",),
    "Flask": (r"\bflask\b",),
    "Django": (r"\bdjango\b",),
    "FastAPI": (r"\bfastapi\b", r"\bfast api\b"),
    "JavaScript": (r"\bjavascript\b", r"\bjs\b"),
    "TypeScript": (r"\btypescript\b", r"\bts\b"),
    "React": (r"\breact(?:\.js)?\b",),
    "Node.js": (r"\bnode(?:\.js)?\b",),
    "MySQL": (r"\bmysql\b",),
    "PostgreSQL": (r"\bpostgres(?:ql)?\b",),
    "MongoDB": (r"\bmongodb\b",),
    "REST APIs": (r"\brest(?:ful)?\s+apis?\b",),
    "Git": (r"\bgit\b",),
    "GitHub": (r"\bgithub\b",),
    "Docker": (r"\bdocker\b",),
    "AWS": (r"\baws\b", r"\bamazon web services\b"),
    "Machine Learning": (r"\bmachine learning\b", r"\bml\b"),
    "Natural Language Processing": (r"\bnatural language processing\b", r"\bnlp\b"),
    "TensorFlow": (r"\btensorflow\b",),
    "PyTorch": (r"\bpytorch\b",),
    "Pandas": (r"\bpandas\b",),
    "NumPy": (r"\bnumpy\b",),
    "Spring Boot": (r"\bspring boot\b",),
    "Hibernate": (r"\bhibernate\b",),
    "Maven": (r"\bmaven\b",),
    "Angular": (r"\bangular\b",),
    "Data Structures": (r"\bdata structures?\b",),
    "Object-Oriented Programming": (r"\bobject[- ]oriented programming\b", r"\boop\b"),
    "C++": (r"\bc\+\+\b",),
    "C#": (r"\bc#\b", r"\bc sharp\b"),
    "OpenCV": (r"\bopencv\b",),
    "Sentence Transformers": (r"\bsentence[- ]transformers?\b", r"\bsentence[- ]bert\b", r"\bsbert\b"),
    "BERT": (r"\bbert\b",),
    "TF-IDF": (r"\btf[- ]idf\b",),
    "Cosine Similarity": (r"\bcosine similarity\b",),
    "scikit-learn": (r"\bscikit[- ]learn\b", r"\bsklearn\b"),
}

_CATEGORIES = {
    "Python": "Programming language", "Java": "Programming language", "C++": "Programming language",
    "C#": "Programming language", "JavaScript": "Programming language", "TypeScript": "Programming language",
    "MySQL": "Database", "PostgreSQL": "Database", "MongoDB": "Database", "SQL": "Database",
    "Flask": "Backend framework", "Django": "Backend framework", "FastAPI": "Backend framework",
    "Node.js": "Backend framework", "Spring Boot": "Backend framework", "REST APIs": "Backend capability",
    "React": "Frontend framework", "Angular": "Frontend framework", "HTML": "Frontend technology", "CSS": "Frontend technology",
    "Machine Learning": "AI/ML", "Natural Language Processing": "NLP", "TensorFlow": "Deep learning",
    "PyTorch": "Deep learning", "OpenCV": "Computer vision", "Sentence Transformers": "NLP",
    "BERT": "NLP", "TF-IDF": "Text representation", "Cosine Similarity": "Text similarity",
    "scikit-learn": "Machine learning library", "Pandas": "Data analysis", "NumPy": "Data analysis",
    "Docker": "DevOps", "AWS": "Cloud", "Git": "Tooling", "GitHub": "Tooling",
}
_ALIASES = {"REST APIs": ("rest api", "restful api"), "Node.js": ("node.js", "nodejs"),
            "Sentence Transformers": ("sentence transformers", "sentence transformer", "sentence-bert", "sbert"),
            "scikit-learn": ("scikit-learn", "sklearn"), "C#": ("c#", "c sharp")}
SKILL_METADATA = {skill: {"category": _CATEGORIES.get(skill, "Technology"),
                          "aliases": _ALIASES.get(skill, (skill.lower(),))}
                  for skill in SKILL_PATTERNS}

# Related technologies contribute partial credit, without claiming that they
# are interchangeable.  This is the local semantic/capability mapping.
CAPABILITY_GROUPS = {
    "database": {"MySQL", "PostgreSQL", "MongoDB", "SQL"},
    "backend": {"Flask", "Django", "FastAPI", "Node.js", "Spring Boot", "REST APIs"},
    "frontend": {"React", "Angular", "JavaScript", "TypeScript", "HTML", "CSS"},
    "machine_learning": {"Machine Learning", "TensorFlow", "PyTorch", "Pandas", "NumPy"},
    "nlp": {"Natural Language Processing", "Sentence Transformers", "BERT", "TF-IDF", "Cosine Similarity"},
    "source_control": {"Git", "GitHub"},
}
PARTIAL_MATCH_CREDIT = 0.60


def extract_skills(text: str) -> list[str]:
    """Return known skills explicitly mentioned in *text*, in stable order."""
    normalized = text.lower()
    return [skill for skill, patterns in SKILL_PATTERNS.items()
            if any(re.search(pattern, normalized) for pattern in patterns)]


def _related_skills(required_skill: str, resume_skills: set[str]) -> list[str]:
    return [skill for group in CAPABILITY_GROUPS.values() if required_skill in group
            for skill in group if skill != required_skill and skill in resume_skills]


def skill_match(job_description: str, resume_text: str) -> dict:
    """Measure coverage of skills explicitly required by the job description."""
    required = extract_skills(job_description)
    resume_skills = set(extract_skills(resume_text))
    matched = [skill for skill in required if skill in resume_skills]
    partial = [{"required_skill": skill, "resume_skill": related[0], "credit": PARTIAL_MATCH_CREDIT}
               for skill in required if skill not in resume_skills
               if (related := _related_skills(skill, resume_skills))]
    partial_requirements = {item["required_skill"] for item in partial}
    missing = [skill for skill in required if skill not in resume_skills and skill not in partial_requirements]
    score = (len(matched) + PARTIAL_MATCH_CREDIT * len(partial)) / len(required) if required else None
    return {"required": required, "matched": matched, "partial": partial, "missing": missing,
            "candidate_skills": sorted(resume_skills), "score": score}
