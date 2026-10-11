# Paper traceability

This project implements the paper *Automated Resume Screening Using Machine Learning: An Ensemble-Based Approach for Efficient Candidate Selection* (Hamid et al., 2026). It does not claim identical results.

| Component | Implementation | Status |
|---|---|---|
| [BASE PAPER] non-alphabetic cleanup, lowercasing, tokenization, stop-word removal, lemmatization | `backend/preprocessing.py` | Section 3.2.2 |
| [BASE PAPER] TF-IDF, `(1,2)` ngrams, 1,000 features | `backend/tfidf_model.py`, `train.py` | Table 2; fitted only on train data |
| [BASE PAPER] LR and Decision Tree benchmarks | `backend/classifier.py`, `backend/evaluation.py` | Table 2 |
| [BASE PAPER] Linear SVM, RF, Multinomial NB | `backend/classifier.py` | Table 2 |
| [BASE PAPER] probability soft voter | `backend/ensemble.py` | Section 3.2.4 |
| [BASE PAPER] 70/30 stratified test split | `train.py` | Table 2 |
| [BASE PAPER] cosine/KNN recommendation and 0.70 threshold | `backend/ranking.py` | Section 3.2.6 |
| [BASE PAPER] browser prototype | `backend/app.py`, `frontend/` | Section 3.2.7 |

The paper also reports a separately curated 598-resume zero-shot corpus and post-hoc fairness audit. Those cannot be reproduced without its raw expert-labelled corpus and audit protocol, so this repository does not invent either dataset or result.
