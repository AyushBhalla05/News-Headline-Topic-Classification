"""
Production Training, Model Selection, Calibration & Evaluation Pipeline.

Strict Academic Methodology:
1. Stratified 80/20 train/test split on genuine real-world news dataset (BBC News).
2. Model selection performed exclusively on training data via 5-Fold Stratified Cross-Validation.
3. Evaluates DummyClassifier, MultinomialNB, LogisticRegression, and LinearSVC.
4. Leakage-safe probability calibration using CalibratedClassifierCV.
5. Single final evaluation on untouched holdout test set (reporting Brier score, Log Loss, and Macro F1).
6. Separate evaluation on independent external benchmark (AG News) for out-of-domain generalization.
7. Separate evaluation on synthetic benchmark for controlled comparison (never combined).
"""

import json
import os
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC

from .preprocess import normalize_headline
from .prepare_datasets import (
    EXTERNAL_DATASET_PATH,
    REAL_DATASET_PATH,
    SYNTHETIC_DATASET_PATH,
    build_external_ag_news_benchmark,
    build_real_world_dataset,
    generate_manifest,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_PATH = MODELS_DIR / "classifier_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
EXTERNAL_METRICS_PATH = MODELS_DIR / "external_metrics.json"
SYNTHETIC_METRICS_PATH = MODELS_DIR / "synthetic_metrics.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


def compute_multiclass_brier_score(y_true, y_probs, classes) -> float:
    """Compute macro-averaged Brier score loss across all classes."""
    y_true_bin = label_binarize(y_true, classes=classes)
    brier_scores = [
        brier_score_loss(y_true_bin[:, i], y_probs[:, i]) for i in range(len(classes))
    ]
    return float(np.mean(brier_scores))


def train_and_evaluate_system():
    print("=" * 70)
    print("PHASE 2.5: ADVANCED ML TRAINING, MODEL SELECTION & EVALUATION")
    print("=" * 70)

    # 1. Ensure datasets and manifest exist
    if not REAL_DATASET_PATH.exists():
        build_real_world_dataset()
    if not EXTERNAL_DATASET_PATH.exists():
        build_external_ag_news_benchmark()
    generate_manifest()

    df_real = pd.read_csv(REAL_DATASET_PATH)
    print(f"Loaded Primary Real-World Dataset: {len(df_real)} samples.")
    print("Class distribution:", df_real["category"].value_counts().to_dict())

    categories = sorted(df_real["category"].unique().tolist())

    # 2. Stratified 80/20 Train / Holdout Test Split
    X = df_real["headline"].values
    y = df_real["category"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train set: {len(X_train)} samples | Untouched Test set: {len(X_test)} samples")

    # 3. Candidate Model Configurations
    vectorizer_params = {
        "preprocessor": normalize_headline,
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "min_df": 2,
        "max_df": 0.90,
    }

    candidates = {
        "Baseline (Dummy)": DummyClassifier(strategy="stratified", random_state=42),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            C=2.5, max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Linear Support Vector (LinearSVC)": LinearSVC(
            C=1.0, class_weight="balanced", random_state=42, dual="auto"
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_benchmark = {}
    fitted_candidate_pipelines = {}

    print("\n--- 5-Fold Cross-Validation on Training Split Exclusively ---")
    for name, clf in candidates.items():
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(**vectorizer_params)),
            ("clf", clf),
        ])

        start_cv = time.perf_counter()
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
        cv_time = time.perf_counter() - start_cv

        # Fit on complete training split
        start_fit = time.perf_counter()
        pipeline.fit(X_train, y_train)
        fit_time = time.perf_counter() - start_fit

        cv_benchmark[name] = {
            "cv_macro_f1_mean": round(float(np.mean(cv_scores)), 4),
            "cv_macro_f1_std": round(float(np.std(cv_scores)), 4),
            "fold_scores": [round(float(s), 4) for s in cv_scores],
            "train_time_sec": round(fit_time, 4),
        }
        fitted_candidate_pipelines[name] = pipeline
        print(f"  {name:35s}: CV Macro F1 = {np.mean(cv_scores):.2%} (+/- {np.std(cv_scores):.2%})")

    # 4. Model Selection exclusively based on Training CV Macro F1
    competitive_models = {k: v for k, v in cv_benchmark.items() if "Baseline" not in k}
    best_candidate_name = max(competitive_models, key=lambda k: competitive_models[k]["cv_macro_f1_mean"])
    print(f"\n--> Model Selected via Training CV: {best_candidate_name} ({cv_benchmark[best_candidate_name]['cv_macro_f1_mean']:.2%})")

    # 5. Leakage-Safe Probability Calibration
    print("\n--- Calibrating Probability Estimates (CalibratedClassifierCV) ---")
    selected_raw_pipeline = fitted_candidate_pipelines[best_candidate_name]

    # Calibrate on training data using 5-fold cross-validation inside the pipeline
    calibrated_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(**vectorizer_params)),
        ("clf", CalibratedClassifierCV(
            estimator=candidates[best_candidate_name],
            method="sigmoid",
            cv=cv
        )),
    ])
    calibrated_pipeline.fit(X_train, y_train)

    # 6. Evaluate on Untouched Holdout Test Set (320 samples)
    print("\n--- Evaluating Winning Model on Untouched Test Set ---")
    y_test_pred = calibrated_pipeline.predict(X_test)
    y_test_probs = calibrated_pipeline.predict_proba(X_test)

    # Compare uncalibrated vs calibrated test metrics
    raw_test_pred = selected_raw_pipeline.predict(X_test)
    if hasattr(selected_raw_pipeline, "predict_proba"):
        raw_test_probs = selected_raw_pipeline.predict_proba(X_test)
        uncalibrated_brier = compute_multiclass_brier_score(y_test, raw_test_probs, categories)
        uncalibrated_logloss = float(log_loss(y_test, raw_test_probs, labels=categories))
    else:
        uncalibrated_brier = None
        uncalibrated_logloss = None

    calibrated_brier = compute_multiclass_brier_score(y_test, y_test_probs, categories)
    calibrated_logloss = float(log_loss(y_test, y_test_probs, labels=categories))

    test_acc = float(accuracy_score(y_test, y_test_pred))
    test_macro_f1 = float(f1_score(y_test, y_test_pred, average="macro"))
    test_weighted_f1 = float(f1_score(y_test, y_test_pred, average="weighted"))
    test_macro_prec = float(precision_score(y_test, y_test_pred, average="macro"))
    test_macro_rec = float(recall_score(y_test, y_test_pred, average="macro"))

    cm = confusion_matrix(y_test, y_test_pred, labels=categories)
    report_dict = classification_report(y_test, y_test_pred, labels=categories, output_dict=True)

    # Single item latency test
    latency_trials = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = calibrated_pipeline.predict([X_test[0]])
        _ = calibrated_pipeline.predict_proba([X_test[0]])
        latency_trials.append((time.perf_counter() - t0) * 1000)
    single_item_latency_ms = round(float(np.mean(latency_trials)), 3)

    print(f"Test Accuracy:           {test_acc:.2%}")
    print(f"Test Macro F1:          {test_macro_f1:.2%}")
    print(f"Calibrated Brier Score: {calibrated_brier:.5f} (Uncalibrated: {uncalibrated_brier})")
    print(f"Calibrated Log Loss:    {calibrated_logloss:.4f} (Uncalibrated: {uncalibrated_logloss})")
    print(f"Mean Inference Latency: {single_item_latency_ms:.3f} ms")

    # 7. Independent External Evaluation (AG News Benchmark)
    print("\n--- Evaluating on Independent External Benchmark (AG News) ---")
    df_ext = pd.read_csv(EXTERNAL_DATASET_PATH)
    X_ext = df_ext["headline"].values
    y_ext = df_ext["category"].values

    y_ext_pred = calibrated_pipeline.predict(X_ext)
    y_ext_probs = calibrated_pipeline.predict_proba(X_ext)

    ext_categories = sorted(df_ext["category"].unique().tolist())
    ext_acc = float(accuracy_score(y_ext, y_ext_pred))
    ext_macro_f1 = float(f1_score(y_ext, y_ext_pred, average="macro"))
    ext_cm = confusion_matrix(y_ext, y_ext_pred, labels=ext_categories)
    ext_report = classification_report(y_ext, y_ext_pred, labels=ext_categories, output_dict=True)
    ext_brier = compute_multiclass_brier_score(y_ext, y_ext_probs, categories)

    print(f"External AG News Accuracy: {ext_acc:.2%}")
    print(f"External AG News Macro F1: {ext_macro_f1:.2%}")

    # 8. Separate Synthetic Benchmark Evaluation (Controlled Comparison)
    print("\n--- Evaluating on Controlled Synthetic Benchmark ---")
    df_syn = pd.read_csv(SYNTHETIC_DATASET_PATH)
    X_syn = df_syn["headline"].values
    y_syn = df_syn["category"].values

    y_syn_pred = calibrated_pipeline.predict(X_syn)
    syn_acc = float(accuracy_score(y_syn, y_syn_pred))
    syn_macro_f1 = float(f1_score(y_syn, y_syn_pred, average="macro"))
    print(f"Synthetic Benchmark Accuracy: {syn_acc:.2%}")

    # 9. Save Artifacts
    # Primary Metrics Payload
    primary_metrics = {
        "selected_model": best_candidate_name,
        "model_selection_criterion": "Highest 5-Fold Stratified CV Macro F1 on Training Split Exclusively",
        "dataset_name": "BBC News Headline Benchmark (Real-World)",
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "categories": categories,
        "cv_results": cv_benchmark,
        "holdout_test_metrics": {
            "accuracy": round(test_acc, 4),
            "macro_f1": round(test_macro_f1, 4),
            "weighted_f1": round(test_weighted_f1, 4),
            "macro_precision": round(test_macro_prec, 4),
            "macro_recall": round(test_macro_rec, 4),
            "log_loss": round(calibrated_logloss, 4),
            "brier_score": round(calibrated_brier, 5),
            "uncalibrated_brier_score": round(uncalibrated_brier, 5) if uncalibrated_brier is not None else None,
            "inference_latency_ms": single_item_latency_ms,
        },
        "confusion_matrix": {
            "labels": categories,
            "matrix": cm.tolist(),
        },
        "per_class_report": {
            cat: {
                "precision": round(report_dict[cat]["precision"], 4),
                "recall": round(report_dict[cat]["recall"], 4),
                "f1_score": round(report_dict[cat]["f1-score"], 4),
                "support": int(report_dict[cat]["support"]),
            }
            for cat in categories
        },
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(primary_metrics, f, indent=2)

    # External Metrics Payload
    external_metrics = {
        "benchmark_name": "AG News Independent Out-of-Domain Evaluation",
        "citation": "Zhang, Zhao, LeCun, NeurIPS 2015",
        "total_samples": len(X_ext),
        "categories": ext_categories,
        "accuracy": round(ext_acc, 4),
        "macro_f1": round(ext_macro_f1, 4),
        "brier_score": round(ext_brier, 5),
        "confusion_matrix": {
            "labels": ext_categories,
            "matrix": ext_cm.tolist(),
        },
        "per_class_report": {
            cat: {
                "precision": round(ext_report[cat]["precision"], 4),
                "recall": round(ext_report[cat]["recall"], 4),
                "f1_score": round(ext_report[cat]["f1-score"], 4),
                "support": int(ext_report[cat]["support"]),
            }
            for cat in ext_categories
        },
    }

    with open(EXTERNAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(external_metrics, f, indent=2)

    # Synthetic Metrics Payload
    synthetic_metrics = {
        "benchmark_name": "Controlled Synthetic / Templated Benchmark",
        "total_samples": len(X_syn),
        "accuracy": round(syn_acc, 4),
        "macro_f1": round(syn_macro_f1, 4),
        "purpose": "Ablation and vocabulary consistency testing (kept separate from real-world claims)",
    }

    with open(SYNTHETIC_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(synthetic_metrics, f, indent=2)

    # Save Pipeline and Metadata
    joblib.dump(calibrated_pipeline, PIPELINE_PATH)
    file_size_kb = round(os.path.getsize(PIPELINE_PATH) / 1024, 2)

    metadata = {
        "model_version": "2.0.0",
        "project_id": "AAI-002",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "selected_architecture": best_candidate_name,
        "calibration": "CalibratedClassifierCV (sigmoid / Platt scaling with 5-fold CV)",
        "vocabulary_size": len(calibrated_pipeline.named_steps["tfidf"].vocabulary_),
        "artifact_size_kb": file_size_kb,
        "primary_dataset": "BBC News Benchmark (real_headlines.csv)",
        "external_benchmark": "AG News (external_ag_news.csv)",
        "categories": categories,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll Phase 2.5 artifacts successfully saved in {MODELS_DIR}:")
    print(f"  - {PIPELINE_PATH.name} ({file_size_kb} KB)")
    print(f"  - {METRICS_PATH.name}")
    print(f"  - {EXTERNAL_METRICS_PATH.name}")
    print(f"  - {SYNTHETIC_METRICS_PATH.name}")
    print(f"  - {METADATA_PATH.name}")


if __name__ == "__main__":
    train_and_evaluate_system()
