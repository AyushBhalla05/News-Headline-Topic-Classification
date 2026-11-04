"""
Model loader and inference service with token-level explainability.
Provides thread-safe prediction execution and probability calculation
for both the production 13-class model and the preserved v1 4-class baseline.
"""

import copy
import json
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib

from ..ml.explainability import ExplainabilityService
from ..ml.preprocess import normalize_headline

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PIPELINE_PATH_V1 = MODELS_DIR / "classifier_pipeline.joblib"
PIPELINE_PATH_13 = MODELS_DIR / "classifier_13class_pipeline.joblib"
METRICS_PATH_V1 = MODELS_DIR / "metrics.json"
METRICS_PATH_13 = MODELS_DIR / "metrics_13class.json"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
EXTERNAL_METRICS_PATH = MODELS_DIR / "external_metrics.json"
SYNTHETIC_METRICS_PATH = MODELS_DIR / "synthetic_metrics.json"


class LRUCache:
    """Thread-safe Least Recently Used (LRU) In-Memory Cache for rapid inference responses."""

    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            self.hits += 1
            self.cache.move_to_end(key)
            return copy.deepcopy(self.cache[key])

    def put(self, key: str, value: Dict[str, Any]):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.capacity:
                    self.cache.popitem(last=False)
            self.cache[key] = copy.deepcopy(value)

    def stats(self) -> Dict[str, Any]:
        with self.lock:
            total = self.hits + self.misses
            hit_ratio = round((self.hits / total) * 100, 2) if total > 0 else 0.0
            return {
                "size": len(self.cache),
                "capacity": self.capacity,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio_pct": hit_ratio
            }

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0


class ModelService:
    def __init__(self):
        self.pipeline = None  # v1 default for backward compatibility
        self.pipeline_13class = None  # Track 1 production 13-class model
        self.lru_cache = LRUCache(capacity=1024)
        self.metrics = {}
        self.metrics_13class = {}
        self.external_metrics = {}
        self.synthetic_metrics = {}
        self.metadata = {}
        self.categories = ["Business", "Entertainment", "Sports", "Technology"]
        self.categories_13 = [
            "Art & Culture", "Business & Economy", "Crime & Law", "Education",
            "Entertainment", "Environment & Climate", "Health & Science",
            "Lifestyle & Society", "Politics", "Sports", "Technology",
            "Women & Gender", "World & International"
        ]
        self.model_name = "CalibratedClassifierCV"
        self.model_version = "2.0.0"
        self.load_artifacts()

    def load_artifacts(self):
        """Load trained pipelines, metadata, and precomputed benchmark metrics."""
        # 1. Load baseline v1 pipeline
        if PIPELINE_PATH_V1.exists():
            self.pipeline = joblib.load(PIPELINE_PATH_V1)
            clf = self.pipeline.named_steps.get("clf")
            self.model_name = clf.__class__.__name__ if clf else "LinearSVC"
            if hasattr(clf, "classes_"):
                self.categories = sorted(list(clf.classes_))

        # 2. Load 13-class Track 1 production pipeline
        if PIPELINE_PATH_13.exists():
            self.pipeline_13class = joblib.load(PIPELINE_PATH_13)
            clf_13 = self.pipeline_13class.named_steps.get("clf")
            if hasattr(clf_13, "classes_"):
                self.categories_13 = sorted(list(clf_13.classes_))

        if METRICS_PATH_V1.exists():
            with open(METRICS_PATH_V1, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

        if METRICS_PATH_13.exists():
            with open(METRICS_PATH_13, "r", encoding="utf-8") as f:
                self.metrics_13class = json.load(f)

        if EXTERNAL_METRICS_PATH.exists():
            with open(EXTERNAL_METRICS_PATH, "r", encoding="utf-8") as f:
                self.external_metrics = json.load(f)

        if SYNTHETIC_METRICS_PATH.exists():
            with open(SYNTHETIC_METRICS_PATH, "r", encoding="utf-8") as f:
                self.synthetic_metrics = json.load(f)

        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
                self.model_version = self.metadata.get("model_version", "2.0.0")

    def is_ready(self) -> bool:
        return self.pipeline is not None or self.pipeline_13class is not None

    def predict(self, headline: str, model_version: str = "auto") -> Dict[str, Any]:
        """
        Execute prediction on the input headline.
        If model_version is '13class' or 'v2', uses 13-class pipeline.
        If model_version is 'v1', uses 4-class pipeline.
        If model_version is 'auto', defaults to v1 when called from v1 endpoints for 100% test preservation.
        """
        if model_version in ["13class", "v2", "track1"]:
            return self.predict_13class(headline)
        return self.predict_v1(headline)

    def predict_v1(self, headline: str) -> Dict[str, Any]:
        """Baseline v1 4-class prediction with token explainability and LRU caching."""
        if self.pipeline is None:
            raise RuntimeError("Baseline v1 model pipeline artifact is missing.")

        start_time = time.perf_counter()
        clean = normalize_headline(headline)

        cache_key = f"v1:{clean}"
        cached = self.lru_cache.get(cache_key)
        if cached is not None:
            cached["headline"] = headline
            cached["inference_time_ms"] = round((time.perf_counter() - start_time) * 1000, 3)
            cached["timestamp"] = datetime.now(timezone.utc).isoformat()
            cached["cache_hit"] = True
            return cached

        pred_category = str(self.pipeline.predict([clean])[0])

        if hasattr(self.pipeline, "predict_proba"):
            raw_probs = self.pipeline.predict_proba([clean])[0]
            classes = [str(c) for c in self.pipeline.classes_]
            probs = {cat: round(float(prob), 4) for cat, prob in zip(classes, raw_probs)}
        else:
            probs = {cat: (1.0 if cat == pred_category else 0.0) for cat in self.categories}

        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        confidence = probs.get(pred_category, 1.0)

        explainability = ExplainabilityService.explain_prediction(
            self.pipeline, clean, pred_category, probs
        )

        result = {
            "headline": headline,
            "category": pred_category,
            "confidence": confidence,
            "probabilities": probs,
            "explainability": explainability,
            "inference_time_ms": latency_ms,
            "model_version": "2.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cache_hit": False
        }
        self.lru_cache.put(cache_key, result)
        return result

    def predict_13class(self, headline: str) -> Dict[str, Any]:
        """Production Track 1 13-class prediction with calibrated probabilities, secondary tag detection, and LRU caching."""
        if self.pipeline_13class is None:
            raise RuntimeError("13-class production model pipeline is not loaded.")
        pipe = self.pipeline_13class

        start_time = time.perf_counter()
        clean = normalize_headline(headline)

        cache_key = f"13class:{clean}"
        cached = self.lru_cache.get(cache_key)
        if cached is not None:
            cached["headline"] = headline
            cached["inference_time_ms"] = round((time.perf_counter() - start_time) * 1000, 3)
            cached["timestamp"] = datetime.now(timezone.utc).isoformat()
            cached["cache_hit"] = True
            return cached

        pred_category = str(pipe.predict([clean])[0])

        if hasattr(pipe, "predict_proba"):
            raw_probs = pipe.predict_proba([clean])[0]
            classes = [str(c) for c in pipe.classes_]
            probs = {cat: round(float(prob), 4) for cat, prob in zip(classes, raw_probs)}
        else:
            probs = {cat: (1.0 if cat == pred_category else 0.0) for cat in self.categories_13}

        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        confidence = probs.get(pred_category, 1.0)

        # Token explainability and standardized ambiguity analysis
        explainability = ExplainabilityService.explain_prediction(
            pipe, clean, pred_category, probs
        )
        ambiguity = explainability.get("ambiguity_analysis", {})
        is_ambiguous = ambiguity.get("is_ambiguous", False)
        margin = ambiguity.get("confidence_margin", 1.0)
        rival_cat = ambiguity.get("competing_category")

        # Secondary tags (explicitly marked as experimental alternative-category estimates)
        secondary_tags = []
        if is_ambiguous and rival_cat:
            secondary_tags.append({
                "category": rival_cat,
                "probability": probs.get(rival_cat, 0.0),
                "margin_to_primary": margin,
                "note": "Experimental alternative-category estimate (Pilot multi-label feature; production classification remains single-label 13-class)"
            })

        result = {
            "headline": headline,
            "category": pred_category,
            "confidence": confidence,
            "probabilities": probs,
            "explainability": explainability,
            "secondary_tags": secondary_tags,
            "is_ambiguous": is_ambiguous,
            "confidence_margin": margin,
            "inference_time_ms": latency_ms,
            "model_version": "3.0.0-13class",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cache_hit": False
        }
        self.lru_cache.put(cache_key, result)
        return result


model_service = ModelService()
