"""Train the paper-reproduction pipeline from the real Kaggle resume dataset."""
from pathlib import Path
import argparse
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from backend.preprocessing import preprocess_text
from backend.tfidf_model import create_vectorizer, save_vectorizer
from backend.classifier import train_classifiers, save_models
from backend.ensemble import build_soft_voting_ensemble
from backend.evaluation import evaluate_models

RANDOM_STATE = 42

def locate_columns(frame: pd.DataFrame) -> tuple[str, str]:
    normalized = {str(c).strip().lower().replace("_", " "): c for c in frame.columns}
    text = next((normalized[k] for k in normalized if k in {"resume", "resume str", "resume text", "resume description"}), None)
    label = next((normalized[k] for k in normalized if k in {"category", "job category", "job_category"}), None)
    if not text or not label:
        raise ValueError(f"Dataset must contain resume text and category columns; found {list(frame.columns)}")
    return text, label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/UpdatedResumeDataSet.csv")
    args = parser.parse_args()
    path = Path(args.dataset)
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}. See data/README.md for the genuine Kaggle dataset setup.")
    raw = pd.read_csv(path)
    text_col, label_col = locate_columns(raw)
    dataset = raw[[text_col, label_col]].dropna().rename(columns={text_col: "resume", label_col: "category"})
    if len(dataset) != 962 or dataset.category.nunique() != 25:
        print(f"Warning: expected paper dataset of 962 resumes/25 categories; got {len(dataset)}/{dataset.category.nunique()}.")
    dataset["cleaned"] = dataset.resume.map(preprocess_text)
    dataset = dataset[dataset.cleaned.str.strip().ne("")]
    X_train_text, X_test_text, y_train, y_test = train_test_split(dataset.cleaned, dataset.category, test_size=.30,
        stratify=dataset.category, random_state=RANDOM_STATE)
    vectorizer = create_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)  # fit only training data: no leakage
    X_test = vectorizer.transform(X_test_text)
    models = train_classifiers(X_train, y_train, RANDOM_STATE)
    ensemble = build_soft_voting_ensemble(models)
    ensemble.fit(X_train, y_train)
    save_vectorizer(vectorizer); save_models(models)
    from backend.classifier import MODEL_DIR
    joblib.dump(ensemble, MODEL_DIR / "soft_voting_ensemble.pkl")
    scores = evaluate_models(models, ensemble, X_test, y_test)
    print(scores.to_string(index=False))
    print("\nSaved models/ and results/ artifacts from actual measurements.")

if __name__ == "__main__": main()
