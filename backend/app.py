"""Flask application for trained resume screening and recommendation."""
from __future__ import annotations
import sys
import logging
from pathlib import Path
from uuid import uuid4
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.preprocessing import preprocess_text
from backend.resume_parser import (extract_candidate_information, extract_resume_text, extract_resumes_from_csv,
                                   ALLOWED_EXTENSIONS)
from backend.tfidf_model import load_vectorizer
from backend.ensemble import prediction_details
from backend.ranking import validate_job_description
from backend.match_explainer import matching_keywords, missing_job_keywords
from backend.jd_analyzer import analyze_job_description
from backend.evidence_extractor import extract_resume_evidence
from backend.evidence_scoring import score_candidate
import joblib

app = Flask(__name__, static_folder=str(ROOT / "frontend"), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
UPLOADS = ROOT / "uploads"; UPLOADS.mkdir(exist_ok=True)
latest_ranking: list[dict] = []
latest_resume_details: dict[str, dict] = {}

@app.after_request
def disable_browser_cache(response):
    """Ensure local development always serves the latest frontend files."""
    response.headers["Cache-Control"] = "no-store"
    return response

def trained_assets():
    vectorizer = load_vectorizer()
    ensemble_path = ROOT / "models" / "soft_voting_ensemble.pkl"
    if not ensemble_path.exists():
        raise FileNotFoundError("No soft-voting ensemble. Run python train.py first.")
    return vectorizer, joblib.load(ensemble_path)


def candidate_failure(filename: str, index: int, stage: str, error: Exception) -> dict:
    """Log an internal error while returning a safe, per-resume UI message."""
    app.logger.exception(
        "[Resume %s] File: %s | Stage: %s | Error: %s: %s",
        index, filename, stage, type(error).__name__, error,
    )
    message = "Could not extract readable text from this resume." if stage == "text_extraction" else "Could not analyze this resume."
    return {"candidate": filename, "index": index, "stage": stage,
            "status": "failed", "reason": message}

@app.get("/")
def index(): return send_from_directory(app.static_folder, "index.html")

@app.get("/health")
def health(): return jsonify(status="ok")

@app.get("/ranking")
def ranking_page(): return send_from_directory(app.static_folder, "ranking.html")

@app.get("/api/ranking")
def get_ranking(): return jsonify(candidates=latest_ranking)

@app.get("/api/resume-details/<detail_id>")
def get_resume_details(detail_id: str):
    details = latest_resume_details.get(detail_id)
    if not details:
        return jsonify(error="Resume details are unavailable. Run screening again."), 404
    return jsonify(details)

@app.errorhandler(413)
def too_large(_): return jsonify(error="Files must be 10 MB or smaller."), 413

@app.post("/api/analyze")
def analyze():
    files = request.files.getlist("resumes")
    bulk_csv = request.files.get("resume_csv")
    job_description = request.form.get("job_description", "")
    debug_mode = request.form.get("debug", "").lower() in {"1", "true", "yes"}
    has_resume_files = any(f.filename for f in files)
    if not has_resume_files and not (bulk_csv and bulk_csv.filename):
        return jsonify(error="Upload at least one resume file or one CSV file."), 400
    if not job_description.strip():
        return jsonify(error="Enter a job description."), 400
    try:
        clean_job = preprocess_text(job_description); validate_job_description(clean_job)
        jd = analyze_job_description(job_description)
        jd["raw_text"] = job_description
        vectorizer, ensemble = trained_assets()
        records, failed_resumes = [], []
        for index, uploaded in enumerate(files, start=1):
            if not uploaded.filename: continue
            filename = secure_filename(uploaded.filename)
            try:
                extension = Path(filename).suffix.lower()
                if extension not in ALLOWED_EXTENSIONS:
                    raise ValueError("Resume file format is not supported. Upload a PDF, DOCX, or TXT file.")
                path = UPLOADS / f"{uuid4().hex}_{filename}"
                uploaded.save(path)
                try:
                    text = extract_resume_text(path)
                except Exception as error:
                    failed_resumes.append(candidate_failure(filename, index, "text_extraction", error))
                    continue
                clean_resume = preprocess_text(text)
                if len(clean_resume.split()) < 5:
                    raise ValueError("Resume contains too little readable text to analyze.")
                records.append({"candidate": filename, "text": text, "cleaned": clean_resume, "index": index})
            except Exception as error:
                failed_resumes.append(candidate_failure(filename, index, "validation", error))
        if bulk_csv and bulk_csv.filename:
            for row_index, row in enumerate(extract_resumes_from_csv(bulk_csv), start=len(files) + 1):
                try:
                    clean_resume = preprocess_text(row["text"])
                    if len(clean_resume.split()) < 5:
                        raise ValueError("Resume contains too little readable text to analyze.")
                    records.append({"candidate": row["candidate"], "text": row["text"], "cleaned": clean_resume, "index": row_index})
                except Exception as error:
                    failed_resumes.append(candidate_failure(row["candidate"], row_index, "validation", error))
        if not records:
            raise ValueError("No usable resume text was found in the uploaded files.")
        job_features = vectorizer.transform([clean_job])
        output = []
        details_by_id = {}
        for record in records:
            try:
                resume_feature = vectorizer.transform([record["cleaned"]])
                classified = prediction_details(ensemble, resume_feature)
                evidence = extract_resume_evidence(record["text"])
                match = score_candidate(jd, evidence, record["text"], vectorizer)
                keywords = matching_keywords(vectorizer, resume_feature, job_features)
                missing_keywords = missing_job_keywords(vectorizer, resume_feature, job_features)
                candidate_info = extract_candidate_information(record["text"], record["candidate"])
                detail_id = uuid4().hex
                details_by_id[detail_id] = {"candidate": candidate_info["name"], "email": candidate_info["email"],
                                         "resume_text": record["text"],
                                         "suggested_role": jd["role"], "role_confidence": jd["role_confidence"],
                                         "role_evidence": jd["role_evidence"], "match_score": match["overall_score"],
                                         "matching": match, "project_analysis": match["projects"],
                                         "project_score": match["project_score"], "semantic_similarity": match["semantic_similarity"],
                                         "text_similarity": match["semantic_similarity"], "semantic_method": match["semantic_method"],
                                         "shared_keywords": keywords, "missing_job_keywords": missing_keywords,
                                         "jd_requirements": jd["requirements"]}
                if debug_mode:
                    details_by_id[detail_id]["debug"] = match["debug"]
                output.append({"candidate": candidate_info["name"], "detail_id": detail_id, "classification": classified,
                           "suggested_role": jd["role"], "role_confidence": jd["role_confidence"],
                           "matching": {**match, "tfidf_similarity": match["semantic_similarity"],
                                        "relevant": match["overall_score"] >= 0.70, "shared_keywords": keywords}})
            except Exception as error:
                failed_resumes.append(candidate_failure(record["candidate"], record["index"], "analysis", error))
        if not output:
            return jsonify(error="None of the uploaded resumes could be analyzed.", failed_resumes=failed_resumes), 400
        output.sort(key=lambda item: (-item["matching"]["overall_score"], -item["matching"]["critical_coverage"],
                                      -(item["matching"]["project_score"] if item["matching"]["project_score"] is not None else -1),
                                      -item["matching"]["semantic_similarity"], item["candidate"].lower()))
        for position, item in enumerate(output, start=1):
            item["rank"] = position
        recommended_count = sum(item["matching"]["relevant"] for item in output)
        global latest_ranking, latest_resume_details
        latest_ranking = [{"candidate": item["candidate"], "rank": item["rank"],
                           "suggested_role": item["suggested_role"],
                           "match_score": item["matching"]["overall_score"],
                           "skill_score": item["matching"]["skill_score"],
                           "project_score": item["matching"]["project_score"],
                           "semantic_similarity": item["matching"]["semantic_similarity"],
                           "critical_coverage": item["matching"]["critical_coverage"],
                           "recommendation": item["matching"]["recommendation"],
                           "recommended": item["matching"]["relevant"]} for item in output]
        latest_resume_details = details_by_id
        return jsonify(candidates=output, recommended_count=recommended_count, failed_resumes=failed_resumes)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        return jsonify(error=str(error)), 400
    except Exception:
        app.logger.exception("Analysis failed")
        return jsonify(error="Analysis failed unexpectedly. Check the server log."), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
