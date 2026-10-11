"""[BASE PAPER] Soft-voting ensemble over SVM, Random Forest and NB."""
from sklearn.ensemble import VotingClassifier

ENSEMBLE_MEMBERS = ("svm", "random_forest", "naive_bayes")

def build_soft_voting_ensemble(models: dict):
    """Create a soft voter from the three paper-selected classifiers."""
    return VotingClassifier(
        estimators=[(name, models[name]) for name in ENSEMBLE_MEMBERS], voting="soft"
    )

def prediction_details(ensemble, features) -> dict:
    probabilities = ensemble.predict_proba(features)[0]
    index = probabilities.argmax()
    return {"category": str(ensemble.classes_[index]), "confidence": float(probabilities[index]),
            "probabilities": {str(label): float(probabilities[i]) for i, label in enumerate(ensemble.classes_)}}
