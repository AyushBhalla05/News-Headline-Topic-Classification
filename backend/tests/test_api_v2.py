"""
Automated Test Suite for Track 1 (13-Category) API v2 Endpoints.
Validates:
1. /api/v2/health: reports 13 active categories, version 3.0.0-13class, and healthy status.
2. /api/v2/categories: returns all 13 categories with descriptions, accents, and samples, plus 2 inactive.
3. /api/v2/predict: classifies headlines across 13 classes with calibrated probabilities and token attributions.
4. /api/v2/benchmarks: returns track1_13class_production metrics alongside baselines.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_v2_health_endpoint():
    response = client.get("/api/v2/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == "3.0.0-13class"
    assert data["active_categories_count"] == 13
    assert len(data["active_categories"]) == 13


def test_v2_categories_endpoint():
    response = client.get("/api/v2/categories")
    assert response.status_code == 200
    data = response.json()
    assert data["total_active"] == 13
    assert len(data["categories"]) == 13
    assert len(data["inactive_reserved"]) == 2

    cat_names = {c["name"] for c in data["categories"]}
    expected = {
        "Art & Culture", "Business & Economy", "Crime & Law", "Education",
        "Entertainment", "Environment & Climate", "Health & Science",
        "Lifestyle & Society", "Politics", "Sports", "Technology",
        "Women & Gender", "World & International"
    }
    assert cat_names == expected

    for cat in data["categories"]:
        assert len(cat["sample_headlines"]) >= 3
        assert cat["color_accent"].startswith("#")


def test_v2_predict_endpoint():
    headline = "Cyberattack Prompts Los Angeles School District To Shut Down Its Computer Systems"
    response = client.post("/api/v2/predict", json={"headline": headline})
    assert response.status_code == 200
    data = response.json()

    assert data["headline"] == headline
    assert data["category"] in [
        "Art & Culture", "Business & Economy", "Crime & Law", "Education",
        "Entertainment", "Environment & Climate", "Health & Science",
        "Lifestyle & Society", "Politics", "Sports", "Technology",
        "Women & Gender", "World & International"
    ]
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 13
    # Check sum of probabilities close to 1
    prob_sum = sum(data["probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.01

    assert "explainability" in data
    assert "top_contributing_tokens" in data["explainability"]
    assert "inference_time_ms" in data
    assert data["model_version"] == "3.0.0-13class"


def test_v2_benchmarks_endpoint():
    response = client.get("/api/v2/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert "track1_13class_production" in data
    assert "baseline_v1_4class" in data
    assert data["track1_13class_production"]["training_metadata"]["categories_count"] == 13
    assert data["track1_13class_production"]["training_metadata"]["training_samples"] == 20800


def test_v2_predict_lru_cache():
    headline = "Global Semiconductor Sales Surge Amid Artificial Intelligence Hardware Demand"
    
    # First request: Cache MISS (runs full model inference)
    res1 = client.post("/api/v2/predict", json={"headline": headline})
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["cache_hit"] is False
    
    # Second request: Cache HIT (served in <1ms from memory)
    res2 = client.post("/api/v2/predict", json={"headline": headline})
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["cache_hit"] is True
    assert data2["category"] == data1["category"]
    assert data2["confidence"] == data1["confidence"]
    assert data2["inference_time_ms"] < 2.0  # Ultra-low latency on cache hit

    # Check health stats reflect cache metrics
    health = client.get("/api/v2/health").json()
    assert "cache_stats" in health
    assert health["cache_stats"]["hits"] >= 1
    assert health["cache_stats"]["size"] >= 1


def test_explainability_uses_fitted_coefficients_not_just_tfidf():
    """Verify explainability uses fitted LinearSVC coefficients and does not fall back to TF-IDF score."""
    headline = "Apple unveils M4 Pro chips with next-gen neural engine accelerators"
    response = client.post("/api/v2/predict", json={"headline": headline})
    assert response.status_code == 200
    data = response.json()

    explain = data["explainability"]
    top_tokens = explain["top_contributing_tokens"]
    assert len(top_tokens) > 0

    # Ensure weights are present (extracted from LinearSVC fold estimators)
    has_weights = any(t.get("weight") is not None for t in top_tokens)
    assert has_weights is True

    # Critical regression check: Fails if attribution merely equals TF-IDF magnitude
    differs_from_tfidf = any(
        t["contribution_score"] != t["tfidf_value"]
        for t in top_tokens
        if t.get("weight") is not None and t["weight"] != 1.0
    )
    assert differs_from_tfidf is True, "Attribution must not merely fall back to TF-IDF magnitude."


def test_ambiguity_logic_standardization_and_consistency():
    """Verify top-level and nested ambiguity outputs are standardized and strictly consistent."""
    headline = "Apple signs multimillion dollar media broadcast rights deal for live sports"
    response = client.post("/api/v2/predict", json={"headline": headline})
    assert response.status_code == 200
    data = response.json()

    nested_ambiguity = data["explainability"]["ambiguity_analysis"]
    # Top-level and nested must be 100% consistent
    assert data["is_ambiguous"] == nested_ambiguity["is_ambiguous"]
    assert data["confidence_margin"] == nested_ambiguity["confidence_margin"]

    # When ambiguous, secondary tags should clarify experimental alternative status
    if data["is_ambiguous"]:
        assert len(data["secondary_tags"]) > 0
        tag = data["secondary_tags"][0]
        assert "Experimental alternative-category estimate" in tag["note"]


def test_database_audit_fields_and_persistence():
    """Verify database records contain latency_ms, model_version, and request_id."""
    headline = "Independent Audit Verification Headline For Persistence Testing"
    req_headers = {"X-Request-ID": "test-audit-req-12345"}
    response = client.post("/api/v2/predict", json={"headline": headline}, headers=req_headers)
    assert response.status_code == 200

    hist_resp = client.get("/history?limit=5")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert hist_data["total"] > 0
    latest = hist_data["items"][0]

    assert "latency_ms" in latest
    assert "model_version" in latest
    assert "request_id" in latest
    assert latest["model_version"] == "3.0.0-13class"


def test_predict_13class_explicit_error_when_pipeline_missing():
    """Verify predict_13class raises explicit RuntimeError if 13-class pipeline is missing (no silent fallback)."""
    from backend.app.main import model_service
    saved = model_service.pipeline_13class
    try:
        model_service.pipeline_13class = None
        with pytest.raises(RuntimeError) as exc_info:
            model_service.predict_13class("Testing headline when model missing")
        assert "13-class production model pipeline is not loaded" in str(exc_info.value)
    finally:
        model_service.pipeline_13class = saved


