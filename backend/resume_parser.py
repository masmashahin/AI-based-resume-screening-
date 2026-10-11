"""Resume text extraction for PDF, DOCX, and TXT uploads."""
from __future__ import annotations
from pathlib import Path
import csv
import io
import re

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def extract_candidate_information(text: str, fallback_name: str) -> dict:
    """Extract a display name and email without ever using a phone number as a name."""
    email_match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, re.IGNORECASE)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Many PDF resumes put the name and headline on one line, separated by an
    # em dash.  Prefer that display name before considering section headings.
    first_line = lines[0] if lines else ""
    heading_name = re.match(r"^([A-Za-z][A-Za-z .'-]{1,70}?)(?:\s+[—–-]\s+|\s*\|)", first_line)
    name = heading_name.group(1).title() if heading_name else next((line for line in lines[:5]
                 if "@" not in line and not re.search(r"\d{5,}", line)
                 and line.lower() not in {"contact", "resume", "summary", "skills"}
                 and re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,70}", line)), None)
    return {"name": name or fallback_name, "email": email_match.group(0) if email_match else None}

def extract_resume_text(path: str | Path) -> str:
    path = Path(path)
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type. Upload a PDF, DOCX, or TXT file.")
    if path.stat().st_size == 0:
        raise ValueError("The uploaded file is empty.")
    try:
        if path.suffix.lower() == ".txt":
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif path.suffix.lower() == ".docx":
            from docx import Document
            text = "\n".join(p.text for p in Document(path).paragraphs)
        else:
            from pypdf import PdfReader
            text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    except Exception as error:
        raise ValueError("Could not read this resume. It may be encrypted, corrupted, or image-only.") from error
    if not text.strip():
        raise ValueError("No selectable text was found in the uploaded resume.")
    return text


def extract_resumes_from_csv(uploaded_file, max_rows: int = 100) -> list[dict]:
    """Read bulk resumes from a UTF-8 CSV with a Resume text column.

    The common Kaggle `Category,Resume` schema is accepted, as is a simple
    `Candidate,Resume` file. Candidate/Name is optional.
    """
    if not uploaded_file.filename.lower().endswith(".csv"):
        raise ValueError("Bulk upload must be a CSV file.")
    try:
        content = uploaded_file.read().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(content))
        if not reader.fieldnames:
            raise ValueError("The CSV has no header row.")
        columns = {name.strip().lower(): name for name in reader.fieldnames if name}
        resume_column = next((columns[key] for key in ("resume", "resume text", "resume_str", "resume description") if key in columns), None)
        name_column = next((columns[key] for key in ("candidate", "name", "candidate name", "filename") if key in columns), None)
        if not resume_column:
            raise ValueError("CSV must contain a Resume column. Download the template if needed.")
        records = []
        for row_number, row in enumerate(reader, start=1):
            text = (row.get(resume_column) or "").strip()
            if not text:
                continue
            candidate = (row.get(name_column) or "").strip() if name_column else ""
            records.append({"candidate": candidate or f"Resume {row_number}", "text": text})
            if len(records) >= max_rows:
                break
    except UnicodeDecodeError as error:
        raise ValueError("CSV must be saved as UTF-8 text.") from error
    except csv.Error as error:
        raise ValueError("Could not read the CSV. Check that commas and quotes are valid.") from error
    if not records:
        raise ValueError("No usable resume text was found in the CSV.")
    return records
