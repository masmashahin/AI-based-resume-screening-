"""[BASE PAPER] TF-IDF factory and persistence helpers."""
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

def create_vectorizer() -> TfidfVectorizer:
    # Paper Table 2 settings.
    return TfidfVectorizer(ngram_range=(1, 2), max_features=1000)

def save_vectorizer(vectorizer: TfidfVectorizer) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)

def load_vectorizer() -> TfidfVectorizer:
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError("No trained vectorizer. Put the dataset in data/ and run python train.py.")
    return joblib.load(VECTORIZER_PATH)
