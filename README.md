# AI-Based Resume Screening and Job Recommendation System

A final-year project implementing the classical ML workflow in Hamid et al. (2026), *Automated Resume Screening Using Machine Learning: An Ensemble-Based Approach for Efficient Candidate Selection*.

## What it does

The system pipeline is: real Kaggle resume dataset → preprocessing → TF-IDF → 70/30 stratified split → SVM, Random Forest, and Multinomial Naive Bayes → probability-based soft voting → category prediction → TF-IDF/cosine KNN resume recommendation.

The UI accepts PDF, DOCX, and TXT resumes plus a job description. Multiple uploaded resumes are ranked using the 0.70 relevance threshold.

## Resume–JD matching

The screening score combines three explainable signals: skill match (50%), project relevance (30%), and resume–JD semantic similarity (20%). These weights, plus the project sub-weights, are configurable in `backend/ranking.py`.

Skills are matched directly and through a small capability map: for example, PostgreSQL is related to MySQL as a database technology, and Django is related to Flask as backend technology. Related skills receive partial credit and are displayed separately from direct matches.

The app also detects project sections under headings such as `Projects`, `Academic Projects`, `Personal Projects`, `Major Project`, and `Project Experience`. Each project is scored from skill coverage (40%), technology overlap (25%), domain similarity (20%), and functional relevance (15%). The strongest two projects are aggregated with a 70/30 emphasis on the strongest one. If no projects are detected, their weight is redistributed across the available skill and semantic signals.

Semantic similarity uses a local `all-MiniLM-L6-v2` Sentence Transformer when that model is already available locally. It never downloads a model while screening. If it is not present, the system uses the existing trained TF-IDF cosine representation and labels this fallback in the match details.

## Dataset and reproducibility

The paper describes 962 resumes across 25 categories but does not identify an exact Kaggle slug. Follow [data setup](data/README.md) to download the genuine `UpdatedResumeDataSet.csv`; no sample or fabricated training data is included.

`train.py` uses random seed 42, a stratified 70/30 split, TF-IDF `(1, 2)` ngrams with `max_features=1000`, Linear SVM `C=0.1` with probability estimates, Random Forest `n_estimators=100, max_depth=13`, and MultinomialNB `alpha=1.2`. It evaluates Logistic Regression, Decision Tree, NB, RF, SVM, and soft voting using actual accuracy, weighted precision/recall/F1, and an ensemble confusion matrix. Results differ with dataset file/version and library environment; none are claimed to equal the paper.

## Installation and training

Use Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Usual NLTK setup (the project also detects data/nltk_data/ if supplied locally)
python -m nltk.downloader stopwords wordnet omw-1.4
python train.py
```

## Run

```bash
python backend/app.py
```

Open `http://127.0.0.1:5000`.

## Artifacts

After training, `models/` contains the TF-IDF vectorizer, standalone models, and soft-voting ensemble. `results/model_comparison.csv`, `results/classification_report.txt`, `results/ensemble_confusion_matrix.csv`, and `results/confusion_matrix.png` contain actual run measurements.

## Limitations

PDF text extraction cannot read image-only PDFs without OCR. The paper’s unpublished 598-resume curated validation corpus and its exact dataset URL are unavailable in the provided manuscript, so their results/audit are not reproduced. This tool supports screening decisions, not automatic hiring decisions; review candidates fairly and keep humans in the loop.

See [PAPER_MAPPING.md](PAPER_MAPPING.md) for exact methodology-to-code traceability.
