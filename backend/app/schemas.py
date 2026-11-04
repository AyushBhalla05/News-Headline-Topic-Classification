"""
Pydantic data schemas for FastAPI request and response validation.
Includes models for prediction, explainability, health, and history.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    headline: str = Field(
        ...,
        min_length=3,
        max_length=300,
        description="The news headline to classify",
        examples=["Manchester City secure dramatic victory in Premier League final"]
    )

    @field_validator("headline")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        clean = v.strip()
        if not clean or len(clean) < 3:
            raise ValueError("Headline cannot be blank or contain fewer than 3 characters.")
        return clean


class TokenAttribution(BaseModel):
    token: str
    tfidf_value: float
    weight: Optional[float] = None
    contribution_score: float


class AmbiguityDetail(BaseModel):
    is_ambiguous: bool
    confidence_margin: float
    competing_category: Optional[str] = None
    explanation: str


class ExplainabilityPayload(BaseModel):
    top_contributing_tokens: List[TokenAttribution]
    opposing_tokens: List[TokenAttribution]
    ambiguity_analysis: AmbiguityDetail
    interpretation_disclaimer: str


class PredictResponse(BaseModel):
    headline: str
    category: str
    confidence: float
    probabilities: Dict[str, float]
    explainability: ExplainabilityPayload
    inference_time_ms: float
    model_version: str
    timestamp: str
    cache_hit: Optional[bool] = Field(default=False, description="Whether prediction was served from in-memory LRU cache")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    model_version: str
    uptime_seconds: float
    supported_categories: List[str]
    cache_stats: Optional[Dict[str, Any]] = None


class CategoryDetail(BaseModel):
    id: str
    name: str
    description: str
    color_accent: str
    sample_headlines: List[str]


class CategoriesResponse(BaseModel):
    total: int
    categories: List[CategoryDetail]


class HistoryItem(BaseModel):
    id: int
    headline: str
    category: str
    confidence: float
    probabilities: Dict[str, float]
    created_at: str
    latency_ms: Optional[float] = None
    model_version: Optional[str] = None
    request_id: Optional[str] = None


class HistoryResponse(BaseModel):
    total: int
    items: List[HistoryItem]
    limit: int
    offset: int
