"""
Hardened FastAPI REST API Test Suite.
Validates:
1. System health, model readiness, and versioning.
2. Categories metadata and UI accents.
3. Inference predictions with calibrated probabilities and token explainability.
4. Input validation and 422 rejection.
5. Multi-benchmark analytics endpoint.
6. Persistence, pagination, filtering, and stable sorting in history.
7. History deletion confirmation safety policy.
8. Request ID tracking and response headers.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.predictor import PIPELINE_PATH_V1

if not PIPELINE_PATH_V1.exists():
    pytest.skip("Baseline v1 4-class model excluded from public release for copyright compliance.", allow_module_level=True)

client = TestClient(app)


def test_health_endpoint():
    """Verify health check returns healthy status, model version, and uptime."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == "2.0.0"
    assert "CalibratedClassifierCV" in data["model_name"] or "LinearSVC" in data["model_name"]
    assert len(data["supported_categories"]) == 4


def test_categories_endpoint():
    """Verify supported categories return metadata, sample headlines, and color accents."""
    response = client.get("/categories")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 4
    names = {c["name"] for c in data["categories"]}
    assert names == {"Sports", "Business", "Entertainment", "Technology"}
    for cat in data["categories"]:
        assert len(cat["sample_headlines"]) >= 3
        assert cat["color_accent"].startswith("#")


def test_predict_endpoint_with_explainability():
    """Verify prediction request returns prediction, probabilities, explainability, and request ID."""
    headline = "Novak Djokovic captures US Open tennis championship"
    response = client.post("/predict", json={"headline": headline})
    assert response.status_code == 200

    data = response.json()
    assert data["headline"] == headline
    assert data["category"] in {"Sports", "Business", "Entertainment", "Technology"}
    assert 0.0 <= data["confidence"] <= 1.0
    assert "probabilities" in data
    assert len(data["probabilities"]) == 4

    # Verify probability distribution sums to 1.0
    assert abs(sum(data["probabilities"].values()) - 1.0) < 1e-3

    # Verify token-level explainability
    assert "explainability" in data
    exp = data["explainability"]
    assert "top_contributing_tokens" in exp
    assert "ambiguity_analysis" in exp
    assert "is_ambiguous" in exp["ambiguity_analysis"]
    assert "interpretation_disclaimer" in exp

    # Verify Request ID headers
    assert "X-Request-ID" in response.headers
    assert "X-Response-Time-Ms" in response.headers


def test_predict_endpoint_input_validation():
    """Verify that blank, short, or oversized headlines are rejected with HTTP 422."""
    # Blank string
    res1 = client.post("/predict", json={"headline": ""})
    assert res1.status_code == 422

    # Whitespace only
    res2 = client.post("/predict", json={"headline": "    "})
    assert res2.status_code == 422

    # Too short (<3 chars)
    res3 = client.post("/predict", json={"headline": "ok"})
    assert res3.status_code == 422

    # Too long (>300 chars)
    res4 = client.post("/predict", json={"headline": "news " * 80})
    assert res4.status_code == 422


def test_benchmarks_endpoint():
    """Verify that all separate benchmarks (real-world, external, synthetic) are served."""
    response = client.get("/benchmarks")
    assert response.status_code == 200
    data = response.json()
    assert "primary_real_world" in data
    assert "independent_external" in data
    assert "controlled_synthetic" in data
    assert "dataset_manifest" in data

    # Verify distinct separate reporting
    assert data["primary_real_world"]["dataset_name"] == "BBC News Headline Benchmark (Real-World)"
    assert data["independent_external"]["benchmark_name"] == "AG News Independent Out-of-Domain Evaluation"
    assert data["controlled_synthetic"]["benchmark_name"] == "Controlled Synthetic / Templated Benchmark"


def test_history_pagination_and_sorting():
    """Verify history persistence, pagination, and stable chronological sorting."""
    # Create two test predictions
    client.post("/predict", json={"headline": "Alpha Headline for History Verification"})
    client.post("/predict", json={"headline": "Beta Headline for History Verification"})

    # Fetch paginated history
    res = client.get("/history?limit=10&offset=0")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] >= 2
    assert len(data["items"]) <= 10

    # Verify stable sorting (newest ID first)
    if len(data["items"]) >= 2:
        assert data["items"][0]["id"] >= data["items"][1]["id"]


def test_destructive_history_deletion_policy():
    """Verify that destructive history deletion is prevented unless explicit confirmation is provided."""
    # Call DELETE without confirmation parameter -> Expect 400 Bad Request
    res_unconfirmed = client.delete("/history")
    assert res_unconfirmed.status_code == 400
    assert "confirm=true" in res_unconfirmed.json()["detail"]

    # Call DELETE with confirm=false -> Expect 400 Bad Request
    res_false = client.delete("/history?confirm=false")
    assert res_false.status_code == 400

    # Call DELETE with confirm=true -> Expect 200 OK
    res_confirmed = client.delete("/history?confirm=true")
    assert res_confirmed.status_code == 200
    assert "cleared successfully" in res_confirmed.json()["message"]

    # Verify history is now empty
    empty_res = client.get("/history")
    assert empty_res.status_code == 200
    assert empty_res.json()["total"] == 0
