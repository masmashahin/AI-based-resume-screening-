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

    def test_jd_role_inference(self):
        jd = analyze_job_description(JD)
        self.assertEqual(jd["role"], "NLP / Machine Learning Engineer")


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


if __name__ == "__main__":
    unittest.main()
