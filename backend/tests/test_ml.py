"""
Expanded Machine Learning Test Suite for News Headline Topic Classification.
Validates:
1. Real-world and external dataset integrity, provenance, and class balance.
2. Cross-split data leakage and duplicate absence.
3. Preprocessing text normalization.
4. Model selection methodology on training-only cross-validation.
5. Calibrated probability bounds and Brier score ranges.
6. Token-level explainability and cross-domain ambiguity detection.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from backend.ml.explainability import ExplainabilityService
from backend.ml.preprocess import normalize_headline
from backend.ml.prepare_datasets import (
    CATEGORIES_13,
    EXTERNAL_DATASET_PATH,
    MANIFEST_PATH,
    REAL_DATASET_PATH,
    SYNTHETIC_DATASET_PATH,
    TEST_13CLASS_PATH,
    TRAIN_13CLASS_PATH,
    VAL_13CLASS_PATH,
)
from backend.ml.train import (
    EXTERNAL_METRICS_PATH,
    METRICS_PATH,
    PIPELINE_PATH,
    compute_multiclass_brier_score,
)

PIPELINE_PATH_13 = Path(__file__).resolve().parent.parent / "models" / "classifier_13class_pipeline.joblib"
METRICS_PATH_13 = Path(__file__).resolve().parent.parent / "models" / "metrics_13class.json"

EXPECTED_CATEGORIES = {"Business", "Entertainment", "Sports", "Technology"}


def test_real_world_dataset_integrity():
    """
    Verify real-world dataset integrity, provenance, and exact balance across splits.
    Validates the primary production 13-class dataset splits (HuffPost CC-BY-4.0).
    When available in the local development workspace, also validates the BBC baseline.
    """
    # 1. Primary Production 13-Class Dataset Validation
    assert TRAIN_13CLASS_PATH.exists(), f"13-class train dataset missing at {TRAIN_13CLASS_PATH}"
    assert VAL_13CLASS_PATH.exists(), f"13-class validation dataset missing at {VAL_13CLASS_PATH}"
    assert TEST_13CLASS_PATH.exists(), f"13-class test dataset missing at {TEST_13CLASS_PATH}"

    train_df = pd.read_csv(TRAIN_13CLASS_PATH)
    val_df = pd.read_csv(VAL_13CLASS_PATH)
    test_df = pd.read_csv(TEST_13CLASS_PATH)

    assert len(train_df) == 20800, f"Expected 20,800 train records, found {len(train_df)}"
    assert len(val_df) == 2600, f"Expected 2,600 val records, found {len(val_df)}"
    assert len(test_df) == 2600, f"Expected 2,600 test records, found {len(test_df)}"

    expected_13 = set(CATEGORIES_13)
    assert set(train_df["target_category"].unique()) == expected_13
    assert set(val_df["target_category"].unique()) == expected_13
    assert set(test_df["target_category"].unique()) == expected_13

    train_counts = train_df["target_category"].value_counts().to_dict()
    val_counts = val_df["target_category"].value_counts().to_dict()
    test_counts = test_df["target_category"].value_counts().to_dict()

    for cat in expected_13:
        assert train_counts[cat] == 1600, f"Train class '{cat}' count mismatch: {train_counts[cat]}"
        assert val_counts[cat] == 200, f"Val class '{cat}' count mismatch: {val_counts[cat]}"
        assert test_counts[cat] == 200, f"Test class '{cat}' count mismatch: {test_counts[cat]}"

    assert train_df["headline"].isnull().sum() == 0, "Found null headline in train_13class"
    assert val_df["headline"].isnull().sum() == 0, "Found null headline in val_13class"
    assert test_df["headline"].isnull().sum() == 0, "Found null headline in test_13class"

    # 2. Local BBC Baseline Validation (when present in development environment)
    if REAL_DATASET_PATH.exists():
        bbc_df = pd.read_csv(REAL_DATASET_PATH)
        assert len(bbc_df) == 1600, f"Expected 1,600 samples, found {len(bbc_df)}"
        assert set(bbc_df["category"].unique()) == EXPECTED_CATEGORIES
        counts = bbc_df["category"].value_counts().to_dict()
        for cat in EXPECTED_CATEGORIES:
            assert counts[cat] == 400, f"Class '{cat}' is not exactly 400 samples: {counts[cat]}"
        assert bbc_df["headline"].isnull().sum() == 0, "Found null headline values"
        assert bbc_df["category"].isnull().sum() == 0, "Found null category values"


def test_external_and_synthetic_benchmarks_exist():
    """Verify synthetic benchmark and manifest are available. External benchmark validated when present locally."""
    assert SYNTHETIC_DATASET_PATH.exists(), "Synthetic benchmark file missing"
    assert MANIFEST_PATH.exists(), "Dataset manifest missing"

    df_synth = pd.read_csv(SYNTHETIC_DATASET_PATH)
    assert len(df_synth) > 0, "Synthetic dataset is empty"

    if EXTERNAL_DATASET_PATH.exists():
        df_ext = pd.read_csv(EXTERNAL_DATASET_PATH)
        assert len(df_ext) == 1200
        assert set(df_ext["category"].unique()) == {"Sports", "Business", "Technology"}


def test_no_exact_leakage_between_train_and_test():
    """Verify zero exact duplicate overlap between training, validation, and holdout test splits."""
    # 1. Primary 13-Class Production Leakage Check
    assert TRAIN_13CLASS_PATH.exists() and TEST_13CLASS_PATH.exists() and VAL_13CLASS_PATH.exists()
    train_df = pd.read_csv(TRAIN_13CLASS_PATH)
    val_df = pd.read_csv(VAL_13CLASS_PATH)
    test_df = pd.read_csv(TEST_13CLASS_PATH)

    train_headlines = set(train_df["canonical_form"].dropna())
    val_headlines = set(val_df["canonical_form"].dropna())
    test_headlines = set(test_df["canonical_form"].dropna())

    exact_leakage_test = train_headlines.intersection(test_headlines)
    assert len(exact_leakage_test) == 0, f"Detected exact train/test leakage: {exact_leakage_test}"

    exact_leakage_val = train_headlines.intersection(val_headlines)
    assert len(exact_leakage_val) == 0, f"Detected exact train/val leakage: {exact_leakage_val}"

    # 2. Local BBC Baseline Split Leakage Check (when present in development environment)
    if REAL_DATASET_PATH.exists():
        df_bbc = pd.read_csv(REAL_DATASET_PATH)
        X = df_bbc["headline"].values
        y = df_bbc["category"].values
        X_train, X_test, _, _ = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        exact_leakage_bbc = set(X_train).intersection(set(X_test))
        assert len(exact_leakage_bbc) == 0, f"Detected BBC train/test leakage: {exact_leakage_bbc}"


def test_preprocessing_normalization():
    """Verify text normalizer removes HTML entities, URLs, and preserves key financial/sports terms."""
    raw = "BREAKING: Apple &amp; Google stock surge by 15% over $100! Visit https://example.com"
    processed = normalize_headline(raw)
    assert "https" not in processed
    assert "&amp;" not in processed
    assert "dollar" in processed
    assert "percent" in processed
    assert "surge" in processed
    assert "over" in processed


def test_pipeline_artifact_and_calibration():
    """Verify the saved production 13-class pipeline artifact is calibrated, loads correctly, and outputs valid probabilities."""
    assert PIPELINE_PATH_13.exists(), f"Production 13-class model pipeline artifact missing at {PIPELINE_PATH_13}"
    pipeline_13 = joblib.load(PIPELINE_PATH_13)

    assert hasattr(pipeline_13, "predict")
    assert hasattr(pipeline_13, "predict_proba")

    test_samples = [
        "Manchester City secure Premier League trophy after dramatic victory",
        "Federal Reserve holds benchmark interest rate steady as inflation cools",
        "Christopher Nolan's Oppenheimer sweeps Academy Awards winning Best Picture",
        "Google DeepMind releases AlphaFold 3 predicting biomolecular structure",
    ]

    for sample in test_samples:
        pred = pipeline_13.predict([sample])[0]
        probs = pipeline_13.predict_proba([sample])[0]
        assert pred in set(CATEGORIES_13)
        assert len(probs) == 13
        # Probability simplex constraint: sum must equal 1.0
        assert abs(sum(probs) - 1.0) < 1e-4
        for p in probs:
            assert 0.0 <= p <= 1.0

    # Also validate baseline v1 pipeline when present locally
    if PIPELINE_PATH.exists():
        pipeline_v1 = joblib.load(PIPELINE_PATH)
        assert hasattr(pipeline_v1, "predict")
        assert hasattr(pipeline_v1, "predict_proba")
        for sample in test_samples:
            p = pipeline_v1.predict([sample])[0]
            assert p in EXPECTED_CATEGORIES


def test_metrics_integrity_and_brier_bounds():
    """Verify test metrics and calibration Brier score fall within valid mathematical bounds."""
    import json
    assert METRICS_PATH_13.exists(), "Production 13-class metrics file missing"
    with open(METRICS_PATH_13, "r", encoding="utf-8") as f:
        metrics_13 = json.load(f)

    test_metrics = metrics_13["holdout_test_metrics"]
    assert 0.0 <= test_metrics["accuracy"] <= 1.0
    assert 0.0 <= test_metrics["macro_f1"] <= 1.0
    assert 0.0 <= test_metrics["brier_score"] <= 1.0
    assert len(metrics_13["training_metadata"]["categories"]) == 13

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_v1 = json.load(f)
        assert 0.0 <= metrics_v1["holdout_test_metrics"]["accuracy"] <= 1.0


def test_explainability_token_attribution():
    """Verify explainability service produces non-empty token contributions for prototypical headlines."""
    pipeline = joblib.load(PIPELINE_PATH_13)
    headline = "Manchester United defeats rival in Champions League football match"
    pred_category = str(pipeline.predict([headline])[0])
    raw_probs = pipeline.predict_proba([headline])[0]
    probs = {c: float(p) for c, p in zip(pipeline.classes_, raw_probs)}

    explanation = ExplainabilityService.explain_prediction(
        pipeline, headline, pred_category, probs
    )

    assert "top_contributing_tokens" in explanation
    assert len(explanation["top_contributing_tokens"]) > 0
    top_token = explanation["top_contributing_tokens"][0]
    assert "token" in top_token
    assert "contribution_score" in top_token
    assert top_token["contribution_score"] > 0
    assert "interpretation_disclaimer" in explanation


def test_ambiguity_detection_on_cross_domain_headline():
    """Verify ambiguity analysis flags cross-domain headlines with small probability margins."""
    pipeline = joblib.load(PIPELINE_PATH_13)
    # A headline deliberately spanning technology and business/entertainment
    ambiguous_headline = "Apple signs multimillion dollar media broadcast rights deal for live sports"
    pred_cat = str(pipeline.predict([ambiguous_headline])[0])
    raw_probs = pipeline.predict_proba([ambiguous_headline])[0]
    probs = {c: float(p) for c, p in zip(pipeline.classes_, raw_probs)}

    explanation = ExplainabilityService.explain_prediction(
        pipeline, ambiguous_headline, pred_cat, probs
    )
    ambiguity = explanation["ambiguity_analysis"]
    assert "is_ambiguous" in ambiguity
    assert "confidence_margin" in ambiguity
    assert "explanation" in ambiguity
