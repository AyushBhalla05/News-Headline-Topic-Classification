"""
Automated Test Suite for Track 2 Multi-Label News Headline Architecture.
Validates:
1. Complete split isolation and disjointness across Track 2 datasets.
2. Absolute isolation of Gold Test Benchmark from all training and validation splits.
3. Multi-hot binary indicator validity (primary=1, indicators in {0, 1}, active tags >= 1).
4. Zero exact and near-duplicate leakage between Track 2 splits (similarity < 0.80).
5. Multi-label model training, inference, and evaluation pipeline integrity (dry-run).
"""

import os
import json
import pytest
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_ML_PATH = os.path.join(DATA_DIR, "train_multilabel.csv")
VAL_ML_PATH = os.path.join(DATA_DIR, "val_multilabel.csv")
TEST_ML_PATH = os.path.join(DATA_DIR, "multilabel_gold_test.csv")
TRAIN_13_PATH = os.path.join(DATA_DIR, "train_13class.csv")
VAL_13_PATH = os.path.join(DATA_DIR, "val_13class.csv")

ACTIVE_CATEGORIES = [
    "Art & Culture",
    "Business & Economy",
    "Crime & Law",
    "Education",
    "Entertainment",
    "Environment & Climate",
    "Health & Science",
    "Lifestyle & Society",
    "Politics",
    "Sports",
    "Technology",
    "Women & Gender",
    "World & International"
]

TAG_COLS = [f"tag_{cat}" for cat in ACTIVE_CATEGORIES]


def test_track2_split_isolation_and_disjointness():
    """Verify that Track 2 Train, Val, and Gold Test datasets have 0 overlapping record IDs."""
    train_df = pd.read_csv(TRAIN_ML_PATH)
    val_df = pd.read_csv(VAL_ML_PATH)
    test_df = pd.read_csv(TEST_ML_PATH)

    assert len(train_df) == 150
    assert len(val_df) == 50
    assert len(test_df) == 50

    s_train = set(train_df["source_record_id"])
    s_val = set(val_df["source_record_id"])
    s_test = set(test_df["source_record_id"])

    assert len(s_train.intersection(s_val)) == 0, "Train-Val overlap in Track 2!"
    assert len(s_train.intersection(s_test)) == 0, "Train-Test overlap in Track 2!"
    assert len(s_val.intersection(s_test)) == 0, "Val-Test overlap in Track 2!"
    assert len(s_train | s_val | s_test) == 250


def test_gold_benchmark_isolation_from_training():
    """Verify that none of the 50 Gold Test Benchmark records appear in any training or tuning split."""
    test_df = pd.read_csv(TEST_ML_PATH)
    gold_ids = set(test_df["source_record_id"])

    train_13 = pd.read_csv(TRAIN_13_PATH)
    val_13 = pd.read_csv(VAL_13_PATH)
    train_ml = pd.read_csv(TRAIN_ML_PATH)
    val_ml = pd.read_csv(VAL_ML_PATH)

    assert len(gold_ids.intersection(set(train_13["source_record_id"]))) == 0, "Gold leaked to train_13class!"
    assert len(gold_ids.intersection(set(val_13["source_record_id"]))) == 0, "Gold leaked to val_13class!"
    assert len(gold_ids.intersection(set(train_ml["source_record_id"]))) == 0, "Gold leaked to train_multilabel!"
    assert len(gold_ids.intersection(set(val_ml["source_record_id"]))) == 0, "Gold leaked to val_multilabel!"


def test_track2_multi_hot_indicator_validity():
    """Verify that every record has valid multi-hot binary indicator vectors."""
    for path, expected_count in [(TRAIN_ML_PATH, 150), (VAL_ML_PATH, 50), (TEST_ML_PATH, 50)]:
        df = pd.read_csv(path)
        assert len(df) == expected_count

        for col in TAG_COLS:
            assert col in df.columns, f"Missing column {col} in {path}"
            # Verify binary values
            unique_vals = set(df[col].unique())
            assert unique_vals.issubset({0, 1}), f"Non-binary values in {col} of {path}"

        for _, row in df.iterrows():
            total_active = sum(row[col] for col in TAG_COLS)
            assert total_active >= 1, f"Record {row['source_record_id']} has 0 active categories!"
            # The primary category must have tag == 1
            primary = row.get("primary_category") or row.get("gold_primary")
            assert row[f"tag_{primary}"] == 1, f"Primary {primary} not marked 1 in record {row['source_record_id']}"


def test_track2_near_duplicate_absence():
    """Verify that Val and Test headlines maintain char 3-gram cosine similarity < 0.80 to Train."""
    train_df = pd.read_csv(TRAIN_ML_PATH)
    val_df = pd.read_csv(VAL_ML_PATH)
    test_df = pd.read_csv(TEST_ML_PATH)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2)
    vec.fit(train_df["headline"].tolist() + val_df["headline"].tolist() + test_df["headline"].tolist())

    X_train = vec.transform(train_df["headline"])
    X_val = vec.transform(val_df["headline"])
    X_test = vec.transform(test_df["headline"])

    sim_val = cosine_similarity(X_val, X_train)
    sim_test = cosine_similarity(X_test, X_train)

    assert sim_val.max() < 0.80, f"Val near-duplicate found in Track 2 (max_sim={sim_val.max():.4f})"
    assert sim_test.max() < 0.80, f"Test near-duplicate found in Track 2 (max_sim={sim_test.max():.4f})"


def test_track2_training_and_evaluation_pipeline_dryrun():
    """Verify that a genuine multi-label classifier can be fitted on train_multilabel and evaluated on gold test."""
    train_df = pd.read_csv(TRAIN_ML_PATH)
    test_df = pd.read_csv(TEST_ML_PATH)

    # Use multi-label columns that exhibit binary class variance in the pilot training set
    active_cols = [c for c in TAG_COLS if 0 < train_df[c].sum() < len(train_df)]

    vec = TfidfVectorizer(max_features=500, stop_words="english")
    X_train = vec.fit_transform(train_df["headline"])
    y_train = train_df[active_cols].values

    X_test = vec.transform(test_df["headline"])
    y_test = test_df[active_cols].values

    # Fit genuine multi-label OneVsRest classifier
    clf = OneVsRestClassifier(LogisticRegression(solver="liblinear", random_state=42))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Calculate standard multi-label metrics
    subset_acc = accuracy_score(y_test, y_pred)
    h_loss = hamming_loss(y_test, y_pred)
    micro_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)

    assert 0.0 <= subset_acc <= 1.0
    assert 0.0 <= h_loss <= 1.0
    assert 0.0 <= micro_f1 <= 1.0
    assert y_pred.shape == y_test.shape
