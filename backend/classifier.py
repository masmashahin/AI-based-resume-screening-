"""[BASE PAPER] Classifier construction, training, prediction, persistence."""
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATHS = {
    "logistic_regression": MODEL_DIR / "logistic_regression_model.pkl",
    "decision_tree": MODEL_DIR / "decision_tree_model.pkl",
    "svm": MODEL_DIR / "svm_model.pkl",
    "random_forest": MODEL_DIR / "random_forest_model.pkl",
    "naive_bayes": MODEL_DIR / "naive_bayes_model.pkl",
}

def build_classifiers(random_state: int = 42):
    """Paper benchmark models and Table 2 configurations."""
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "decision_tree": DecisionTreeClassifier(max_depth=15, random_state=random_state),
        # SVC rather than LinearSVC is intentional: soft voting requires predict_proba.
        "svm": SVC(kernel="linear", C=0.1, probability=True, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=13, random_state=random_state, n_jobs=-1),
        "naive_bayes": MultinomialNB(alpha=1.2),
    }

def train_classifiers(X_train, y_train, random_state: int = 42):
    models = build_classifiers(random_state)
    for model in models.values():
        model.fit(X_train, y_train)
    return models

def save_models(models: dict) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    for name, model in models.items():
        joblib.dump(model, MODEL_PATHS[name])

def load_models() -> dict:
    missing = [str(path.name) for path in MODEL_PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing trained model(s): " + ", ".join(missing))
    return {name: joblib.load(path) for name, path in MODEL_PATHS.items()}
