"""[BASE PAPER] Actual-measurement evaluation and result artifacts."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

def evaluate_models(models: dict, ensemble, X_test, y_test) -> pd.DataFrame:
    rows, reports = [], []
    all_models = {**models, "soft_voting_ensemble": ensemble}
    for name, model in all_models.items():
        predicted = model.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predicted, average="weighted", zero_division=0)
        rows.append({"model": name, "accuracy": accuracy_score(y_test, predicted), "precision_weighted": precision,
                     "recall_weighted": recall, "f1_weighted": f1})
        reports.append(f"\n{'=' * 72}\n{name}\n" + classification_report(y_test, predicted, zero_division=0))
    RESULTS_DIR.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    (RESULTS_DIR / "classification_report.txt").write_text("\n".join(reports), encoding="utf-8")
    # Store both a data artifact and the visual confusion matrix requested for the project.
    matrix = confusion_matrix(y_test, ensemble.predict(X_test))
    pd.DataFrame(matrix, index=ensemble.classes_, columns=ensemble.classes_).to_csv(RESULTS_DIR / "ensemble_confusion_matrix.csv")
    # Pillow avoids a system font-cache dependency and makes this artifact portable.
    from PIL import Image, ImageDraw
    cell, margin = 24, 180
    size = margin + cell * len(ensemble.classes_)
    image = Image.new("RGB", (size + 30, size + 30), "white")
    draw = ImageDraw.Draw(image)
    maximum = max(int(matrix.max()), 1)
    for row in range(len(ensemble.classes_)):
        for column in range(len(ensemble.classes_)):
            value = int(matrix[row, column])
            intensity = int(255 - 190 * (value / maximum))
            x, y = margin + column * cell, margin + row * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=(intensity, intensity + min(40, 255-intensity), 255), outline="#d0d7e2")
            if value:
                draw.text((x + 5, y + 5), str(value), fill="black")
        draw.text((4, margin + row * cell + 5), str(ensemble.classes_[row])[:24], fill="black")
    for column, label in enumerate(ensemble.classes_):
        draw.text((margin + column * cell + 3, 5), str(label)[:3], fill="black")
    draw.text((4, 4), "Actual / Predicted", fill="black")
    image.save(RESULTS_DIR / "confusion_matrix.png")
    return frame
