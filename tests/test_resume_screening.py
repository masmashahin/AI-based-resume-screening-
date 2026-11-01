"""Regression tests for upload isolation and evidence-weighted matching."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from docx import Document
from pypdf import PdfWriter

from backend.app import app
from backend.evidence_extractor import extract_resume_evidence
from backend.jd_analyzer import analyze_job_description
from backend.resume_parser import extract_resume_text
from backend.skill_matcher import extract_skills


JD = ("NLP Engineer required. Required Python, Natural Language Processing, "
      "Machine Learning and Sentence Transformers. Preferred Docker.")

PYTHON_DEVELOPER_JD = """Python Developer
Job Responsibilities:
- Develop REST APIs and debug production issues.
Required Skills:
- Python, OOP, SQL, JSON, Git, and Unit Testing with pytest.
- Any one of Flask, Django or FastAPI.
Qualifications:
- Debugging is essential.
Preferred Skills:
- Docker is nice to have.
"""


class ResumeExtractionTests(unittest.TestCase):
    def test_txt_and_docx_extraction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            text_file = root / "resume.txt"
            text_file.write_text("Jane Doe\nPython NLP", encoding="utf-8")
            docx_file = root / "resume.docx"
            doc = Document(); doc.add_paragraph("Jane Doe Python NLP"); doc.save(docx_file)
            self.assertIn("Python", extract_resume_text(text_file))
            self.assertIn("Python", extract_resume_text(docx_file))

    def test_empty_pdf_is_reported(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "empty.pdf"
            writer = PdfWriter(); writer.add_blank_page(width=72, height=72)
            with path.open("wb") as handle: writer.write(handle)
            with self.assertRaisesRegex(ValueError, "No selectable text"):
                extract_resume_text(path)


class EvidenceTests(unittest.TestCase):
    def test_exact_and_related_capabilities_are_identified(self):
        self.assertIn("Python", extract_skills("Python and MySQL"))
        evidence = extract_resume_evidence("Skills\nPython, MySQL\nProjects\nData API — Python and MySQL.")
        self.assertIn("Python", evidence["candidate_skills"])

    def test_negative_skill_statement_is_not_evidence(self):
        evidence = extract_resume_evidence("Experience\nNo demonstrated NLP, BERT, or Sentence Transformers experience.")
        self.assertNotIn("BERT", evidence["candidate_skills"])
        self.assertNotIn("Natural Language Processing", evidence["candidate_skills"])

    def test_context_only_skill_mentions_are_not_evidence(self):
        text = ("Summary\nVisual design rather than Python application or backend development.\n"
                "No experience in Flask. Looking to learn Django. Python is required for this job.\n"
                "Experience\nExperienced in SQL, but no Python experience.")
        evidence = extract_resume_evidence(text)
        self.assertNotIn("Python", evidence["candidate_skills"])
        self.assertNotIn("Flask", evidence["candidate_skills"])
        self.assertNotIn("Django", evidence["candidate_skills"])
        self.assertIn("SQL", evidence["candidate_skills"])

    def test_demonstrated_skill_mentions_remain_evidence(self):
        text = "Experience\nDeveloped applications using Python. Built REST APIs using Flask."
        evidence = extract_resume_evidence(text)
        self.assertIn("Python", evidence["candidate_skills"])
        self.assertIn("Flask", evidence["candidate_skills"])

    def test_jd_role_inference(self):
        jd = analyze_job_description(JD)
        self.assertEqual(jd["role"], "NLP / Machine Learning Engineer")

    def test_section_aware_jd_requirements_and_framework_alternatives(self):
        jd = analyze_job_description(PYTHON_DEVELOPER_JD)
        requirements = {item["skill"]: item for item in jd["requirements"]}
        self.assertEqual(len(requirements), len(jd["requirements"]))
        for skill in ("Python", "Object-Oriented Programming", "REST APIs", "SQL", "JSON", "Git", "Unit Testing", "Debugging"):
            self.assertIn(skill, requirements)
        group = requirements["Flask / Django / FastAPI"]
        self.assertEqual(group["alternatives"], ["Flask", "Django", "FastAPI"])
        self.assertEqual(requirements["Python"]["requirement_type"], "required")
        self.assertTrue(requirements["Python"]["critical"])
        self.assertEqual(requirements["Docker"]["requirement_type"], "preferred")
        self.assertTrue(requirements["Debugging"]["critical"])

    def test_alternative_framework_group_receives_credit_for_one_member(self):
        jd = analyze_job_description("Required Skills: Flask / Django / FastAPI")
        from backend.evidence_scoring import score_requirements
        analyses, skill_score, _, _ = score_requirements(
            jd["requirements"], {"entries": [{"skill": "Django", "source": "skills", "context": "", "project_name": None}]}
        )
        self.assertEqual(len(analyses), 1)
        self.assertEqual(analyses[0]["match_strength"], 1.0)
        self.assertGreater(skill_score, 0.0)

    def test_negative_jd_mentions_do_not_create_requirements(self):
        jd = analyze_job_description("Required Skills: Python. Java is not required. No Docker experience is needed.")
        self.assertEqual([item["skill"] for item in jd["requirements"]], ["Python"])

    def test_no_critical_requirements_reports_unavailable_coverage(self):
        from backend.evidence_scoring import score_requirements
        jd = analyze_job_description("Job Responsibilities:\n- Develop Python services.")
        _, _, _, critical_coverage = score_requirements(
            jd["requirements"], {"entries": [{"skill": "Python", "source": "skills", "context": "", "project_name": None}]}
        )
        self.assertIsNone(critical_coverage)

    def test_critical_coverage_is_zero_without_critical_evidence(self):
        from backend.evidence_scoring import score_requirements
        jd = analyze_job_description("Required Skills: Python. Debugging is essential.")
        _, _, _, critical_coverage = score_requirements(
            jd["requirements"], {"entries": [{"skill": "SQL", "source": "skills", "context": "", "project_name": None}]}
        )
        self.assertEqual(critical_coverage, 0.0)

    def test_multiple_critical_requirements_are_weighted(self):
        from backend.evidence_scoring import score_requirements
        jd = analyze_job_description("Python is essential. Debugging is essential.")
        _, _, _, critical_coverage = score_requirements(
            jd["requirements"], {"entries": [{"skill": "Debugging", "source": "skills", "context": "", "project_name": None}]}
        )
        self.assertGreater(critical_coverage, 0.0)
        self.assertLess(critical_coverage, 1.0)

    def test_debugged_is_debugging_evidence(self):
        evidence = extract_resume_evidence("Experience\nDebugged production Python services.")
        self.assertIn("Debugging", evidence["candidate_skills"])


class AnalyzeEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_missing_inputs_are_useful_errors(self):
        self.assertEqual(self.client.post("/api/analyze", data={}).status_code, 400)
        response = self.client.post("/api/analyze", data={"job_description": "Python NLP role"})
        self.assertEqual(response.status_code, 400)

    def test_one_broken_resume_does_not_abort_batch(self):
        good = ("Jane Doe\nSkills\nPython, NLP, Machine Learning\nProjects\n"
                "Text Classifier — Built an NLP classifier using Python.")
        response = self.client.post("/api/analyze", data={
            "job_description": JD,
            "resumes": [(BytesIO(b"not a pdf"), "broken.pdf"),
                        (BytesIO(good.encode()), "good.txt")],
        }, content_type="multipart/form-data")
        data = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(data["candidates"]), 1)
        self.assertEqual(data["failed_resumes"][0]["stage"], "text_extraction")

    def test_python_developer_jd_reaches_scoring_with_all_requirements(self):
        resume = ("Avery Dev\nSkills\nPython, SQL, JSON, Git, pytest, Django\n"
                  "Experience\nBuilt REST APIs and debugged production services using OOP.")
        response = self.client.post("/api/analyze", data={
            "job_description": PYTHON_DEVELOPER_JD,
            "resumes": [(BytesIO(resume.encode()), "developer.txt")],
        }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        candidate = response.get_json()["candidates"][0]
        requirements = candidate["matching"]["requirements"]
        self.assertEqual(len(requirements), 10)  # Includes preferred Docker.
        self.assertIn("Flask / Django / FastAPI", [item["skill"] for item in requirements])
        self.assertIn("Debugging", [item["skill"] for item in requirements])
        self.assertLess(candidate["matching"]["skill_score"], 1.0)
        details = self.client.get(f"/api/resume-details/{candidate['detail_id']}")
        self.assertEqual(details.status_code, 200)
        self.assertIn("matching", details.get_json())


if __name__ == "__main__":
    unittest.main()
