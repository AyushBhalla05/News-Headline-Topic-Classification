"""
Training and Evaluation Pipeline for Track 1 (13-Category Primary Topic Classifier).
Source Data:
- Train: backend/data/train_13class.csv (20,800 records, 1,600/class)
- Val:   backend/data/val_13class.csv   (2,600 records, 200/class)
- Test:  backend/data/test_13class.csv  (2,600 records, 200/class)

Pipeline:
- Clean Text Normalizer
- TF-IDF N-gram Feature Extraction (unigrams + bigrams, sublinear TF)
- Calibrated Linear Support Vector Classifier (5-Fold Cross-Validation Platt Scaling)
- Rigorous Multi-Class Calibration (Multi-Class Brier Score)
- Evaluation against Holdout Test Set (2,600 records)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    brier_score_loss,
)
from backend.ml.preprocess import normalize_headline

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_OUTPUT_PATH = os.path.join(MODEL_DIR, "classifier_13class_pipeline.joblib")
METRICS_OUTPUT_PATH = os.path.join(MODEL_DIR, "metrics_13class.json")

TRAIN_PATH = os.path.join(DATA_DIR, "train_13class.csv")
VAL_PATH = os.path.join(DATA_DIR, "val_13class.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_13class.csv")


def load_datasets():
    print("Loading 13-class datasets...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"Loaded {len(train_df)} train, {len(val_df)} val, {len(test_df)} test records.")
    return train_df, val_df, test_df


def compute_multiclass_brier_score(y_true, y_prob, classes):
    """Computes multi-class Brier score = mean sum((p_ic - y_ic)^2)."""
    n_samples = len(y_true)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true_indices = np.array([class_to_idx[c] for c in y_true])
    y_one_hot = np.zeros((n_samples, len(classes)))
    y_one_hot[np.arange(n_samples), y_true_indices] = 1.0

    brier = np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1))
    return float(brier)


def train_and_evaluate():
    train_df, val_df, test_df = load_datasets()

    # Preprocess text
    print("Preprocessing headlines...")
    X_train = [normalize_headline(h) for h in train_df["headline"]]
    y_train = train_df["target_category"].values

    X_val = [normalize_headline(h) for h in val_df["headline"]]
    y_val = val_df["target_category"].values

    X_test = [normalize_headline(h) for h in test_df["headline"]]
    y_test = test_df["target_category"].values

    classes = sorted(list(set(y_train)))
    print(f"Target categories ({len(classes)}): {classes}")

    # Build Pipeline: TF-IDF + Calibrated LinearSVC
    print("\nConfiguring pipeline: TF-IDF + CalibratedClassifierCV(LinearSVC)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
        norm="l2"
    )

    base_svc = LinearSVC(C=0.8, penalty="l2", loss="squared_hinge", random_state=42, max_iter=2000)
    calibrated_clf = CalibratedClassifierCV(estimator=base_svc, method="sigmoid", cv=5)

    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", calibrated_clf)
    ])

    # Fit pipeline
    start_time = time.time()
    print("Fitting model on 20,800 training examples with 5-fold internal calibration...")
    pipeline.fit(X_train, y_train)
    training_duration = round(time.time() - start_time, 2)
    print(f"Training completed in {training_duration}s.")

    # Validation Evaluation
    val_pred = pipeline.predict(X_val)
    val_prob = pipeline.predict_proba(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_macro_f1 = f1_score(y_val, val_pred, average="macro")
    val_brier = compute_multiclass_brier_score(y_val, val_prob, pipeline.classes_)
    print(f"\nValidation Performance (2,600 records):")
    print(f"  Accuracy: {val_acc*100:.2f}% | Macro F1: {val_macro_f1*100:.2f}% | Brier Score: {val_brier:.4f}")

    # Test Evaluation
    test_pred = pipeline.predict(X_test)
    test_prob = pipeline.predict_proba(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_macro_prec = precision_score(y_test, test_pred, average="macro")
    test_macro_rec = recall_score(y_test, test_pred, average="macro")
    test_macro_f1 = f1_score(y_test, test_pred, average="macro")
    test_weighted_f1 = f1_score(y_test, test_pred, average="weighted")
    test_weighted_prec = precision_score(y_test, test_pred, average="weighted")
    test_weighted_rec = recall_score(y_test, test_pred, average="weighted")
    test_brier = compute_multiclass_brier_score(y_test, test_prob, pipeline.classes_)
    cm = confusion_matrix(y_test, test_pred, labels=pipeline.classes_).tolist()
    cls_report = classification_report(y_test, test_pred, labels=pipeline.classes_, output_dict=True)

    print(f"\nTest Performance (2,600 held-out records):")
    print(f"  Accuracy:     {test_acc*100:.2f}%")
    print(f"  Macro F1:     {test_macro_f1*100:.2f}%")
    print(f"  Weighted F1:  {test_weighted_f1*100:.2f}%")
    print(f"  Macro Rec:    {test_macro_rec*100:.2f}%")
    print(f"  Macro Pre:    {test_macro_prec*100:.2f}%")
    print(f"  Brier Score:  {test_brier:.4f}")

    # Save Pipeline Artifact
    joblib.dump(pipeline, MODEL_OUTPUT_PATH)
    file_size_kb = round(os.path.getsize(MODEL_OUTPUT_PATH) / 1024, 2)
    print(f"\nSaved trained 13-class pipeline to {MODEL_OUTPUT_PATH} ({file_size_kb} KB)")

    # Metrics JSON
    metrics_data = {
        "model_version": "3.0.0-13class-track1",
        "model_architecture": "TF-IDF(ngram=(1,2), min_df=2) + CalibratedClassifierCV(LinearSVC(C=0.8), cv=5, method='sigmoid')",
        "training_metadata": {
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
            "categories_count": len(classes),
            "categories": list(pipeline.classes_),
            "training_time_seconds": training_duration,
            "artifact_size_kb": file_size_kb,
            "trained_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        },
        "validation_metrics": {
            "accuracy": round(float(val_acc), 4),
            "macro_f1": round(float(val_macro_f1), 4),
            "brier_score": round(float(val_brier), 4)
        },
        "holdout_test_metrics": {
            "accuracy": round(float(test_acc), 4),
            "macro_precision": round(float(test_macro_prec), 4),
            "macro_recall": round(float(test_macro_rec), 4),
            "macro_f1": round(float(test_macro_f1), 4),
            "weighted_precision": round(float(test_weighted_prec), 4),
            "weighted_recall": round(float(test_weighted_rec), 4),
            "weighted_f1": round(float(test_weighted_f1), 4),
            "brier_score": round(float(test_brier), 4),
            "confusion_matrix": cm,
            "classes": list(pipeline.classes_),
            "per_class_metrics": {
                c: {
                    "precision": round(cls_report[c]["precision"], 4),
                    "recall": round(cls_report[c]["recall"], 4),
                    "f1_score": round(cls_report[c]["f1-score"], 4),
                    "support": cls_report[c]["support"]
                }
                for c in pipeline.classes_
            }
        },
        "baseline_comparison": {
            "v1_baseline_categories": 4,
            "v1_random_chance": 0.25,
            "v1_holdout_accuracy": 0.7937,
            "v1_macro_f1": 0.7933,
            "track1_categories": 13,
            "track1_random_chance": 0.0769,
            "track1_holdout_accuracy": round(float(test_acc), 4),
            "track1_macro_f1": round(float(test_macro_f1), 4),
            "lift_over_random_chance": f"{round(float(test_acc) / 0.076923, 2)}x (Track 1) vs 3.17x (v1 Baseline)"
        }
    }

    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Saved metrics to {METRICS_OUTPUT_PATH}")

    return metrics_data


if __name__ == "__main__":
    train_and_evaluate()
