"""
FastAPI Main Application for News Headline Topic Classification Platform.
Production-hardened REST API with dual-track support:
- Track 1 Production: 13-Category Topic Classification with probability calibration, token explainability, and ambiguity analysis.
- Baseline v1: 4-Category BBC benchmark preservation.
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from .database import PredictionRecord, get_db, init_db
from .predictor import model_service
from .schemas import (
    CategoriesResponse,
    CategoryDetail,
    HealthResponse,
    HistoryItem,
    HistoryResponse,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger("news_classifier_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)

START_TIME = time.time()
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MANIFEST_PATH = DATA_DIR / "dataset_manifest.json"
MANIFEST_15_PATH = DATA_DIR / "dataset_manifest_15class.json"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing SQLite persistence layer...")
    init_db()
    logger.info("Validating model pipeline readiness...")
    if model_service.is_ready():
        logger.info(f"Model service ready: v1={model_service.pipeline is not None}, 13class={model_service.pipeline_13class is not None}")
    else:
        logger.warning("Model pipeline artifacts not found at startup.")
    yield
    logger.info("Application shutting down.")


app = FastAPI(
    title="News Intelligence Platform API",
    description="Scalable News Topic Classification API with probability calibration, token explainability, and cross-domain ambiguity detection.",
    version="3.0.0",
    lifespan=lifespan,
)

# Request ID Middleware
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = req_id
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Response-Time-Ms"] = str(duration_ms)
    logger.info(f"{request.method} {request.url.path} [{response.status_code}] ({duration_ms}ms) req_id={req_id}")
    return response


# Explicit CORS Configuration
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "https://newsintel-v3.vercel.app",
    "https://frontend-pi-lovat-mn8yblon30.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)

# 4-class metadata preserved for v1 baseline
CATEGORIES_METADATA_V1 = [
    CategoryDetail(
        id="sports",
        name="Sports",
        description="Headlines covering professional athletics, tournaments, leagues, matches, and player transfers.",
        color_accent="#10B981",
        sample_headlines=[
            "Manchester City secure Premier League title after dramatic final day comeback",
            "Novak Djokovic clinches record 24th Grand Slam singles title at US Open",
            "Max Verstappen dominates Monaco Grand Prix in changing weather conditions",
            "India defeats South Africa in thrilling T20 World Cup final to lift ICC trophy"
        ]
    ),
    CategoryDetail(
        id="business",
        name="Business",
        description="Headlines covering global markets, corporate finance, interest rates, macroeconomic trends, and investments.",
        color_accent="#3B82F6",
        sample_headlines=[
            "Federal Reserve holds interest rates steady as inflation shows signs of cooling",
            "Nvidia briefly surpasses Microsoft as most valuable publicly traded company in the world",
            "Wall Street closes higher led by rally in semiconductor and banking shares",
            "Treasury yields rise sharply following unexpectedly strong monthly jobs report"
        ]
    ),
    CategoryDetail(
        id="entertainment",
        name="Entertainment",
        description="Headlines covering cinema, box office records, music, awards, television series, and celebrity culture.",
        color_accent="#EC4899",
        sample_headlines=[
            "Christopher Nolan's Oppenheimer sweeps Academy Awards winning seven Oscars including Best Picture",
            "Taylor Swift's Eras Tour becomes highest-grossing concert tour in music history",
            "Inside Out 2 crosses 1.5 billion dollars at global box office to become top animated film",
            "Marvel Studios unveils Robert Downey Jr returning to MCU as Doctor Doom in upcoming Avengers"
        ]
    ),
    CategoryDetail(
        id="technology",
        name="Technology",
        description="Headlines covering artificial intelligence, frontier hardware, semiconductors, cybersecurity, and space exploration.",
        color_accent="#8B5CF6",
        sample_headlines=[
            "OpenAI unveils GPT-4o with real-time audio, vision and multilingual voice capabilities",
            "Google DeepMind releases AlphaFold 3 predicting structure of all life molecules and interactions",
            "CrowdStrike software update causes global IT outage grounding flights and disrupting banks",
            "TSMC starts construction on cutting-edge 2nm semiconductor fabrication plant in Taiwan"
        ]
    ),
]

# 13-class metadata for Track 1 production
CATEGORIES_METADATA_13 = [
    CategoryDetail(
        id="art-culture",
        name="Art & Culture",
        description="Fine arts, literature, exhibitions, architecture, poetry, and cultural heritage.",
        color_accent="#8B5CF6",
        sample_headlines=[
            "A Poetry App Worth a Look and a Listen",
            "14 Iranian Artists Explore Just How Complex Immigration and Identity Can Be",
            "That Which Might Have Been: 50 Year Anniversary Exhibition Commemoration"
        ]
    ),
    CategoryDetail(
        id="business-economy",
        name="Business & Economy",
        description="Corporate finance, startups, stock markets, inflation, labor unions, and commerce.",
        color_accent="#3B82F6",
        sample_headlines=[
            "Federal Reserve Holds Interest Rates Steady Amid Cooling Inflation Data",
            "Founders Finding Funding From Friends May Face Regulatory Hurdles",
            "Positive Customer Experience: What is the Real Return on Investment"
        ]
    ),
    CategoryDetail(
        id="crime-law",
        name="Crime & Law",
        description="Criminal investigations, court verdicts, law enforcement, judicial trials, and police reform.",
        color_accent="#6366F1",
        sample_headlines=[
            "Thief Dresses Like Apple Store Employee and Makes Off With 19 iPhones",
            "Colorado Sheriff Indicted on Charges of Inmate Civil Rights Violations",
            "State Supreme Court Issues Landmark Decision on Corporate Trade Secret Law"
        ]
    ),
    CategoryDetail(
        id="education",
        name="Education",
        description="K-12 schools, universities, tuition policy, academic research, and student governance.",
        color_accent="#F59E0B",
        sample_headlines=[
            "Cyberattack Prompts Los Angeles School District To Shut Down Its Computer Systems",
            "Intersecting Ideas: The Importance of an Interdisciplinary University Education",
            "Teachers in Ohio Largest School District Reach Agreement After Extended Strike"
        ]
    ),
    CategoryDetail(
        id="entertainment",
        name="Entertainment",
        description="Cinema, television, comedy, late-night satire, music albums, and pop culture.",
        color_accent="#EC4899",
        sample_headlines=[
            "Cardi B Donates 100,000 Dollars To Her Old Middle School in The Bronx",
            "The Best Late Night Political Comedy of The Week and Satirical Monologues",
            "Mel Brooks American Master Retrospective Celebrates Legendary Comedy Career"
        ]
    ),
    CategoryDetail(
        id="environment-climate",
        name="Environment & Climate",
        description="Climate research, global warming, renewable energy, ocean ecosystems, and wildlife.",
        color_accent="#22C55E",
        sample_headlines=[
            "Collapse of Atlantic Ocean Current Could Trigger Icy Apocalypse Researchers Warn",
            "Arctic Warming Made Monster Storm a Severe Meteorological Catastrophe",
            "Solar and Wind Generation Surpasses Fossil Fuel Production Across European Grid"
        ]
    ),
    CategoryDetail(
        id="health-science",
        name="Health & Science",
        description="Biomedical breakthroughs, neuroscience, pathology, vaccines, and public health clinical studies.",
        color_accent="#14B8A6",
        sample_headlines=[
            "Study Explains The Exact Neurobiological Reasons Why Some Movies Are So Scary",
            "Clinical Trial Results Confirm Significant Efficacy for Novel Cancer Immunotherapy",
            "A Hidden Danger and Side Effect Identified in Modern Formulations of Prescription Drugs"
        ]
    ),
    CategoryDetail(
        id="lifestyle-society",
        name="Lifestyle & Society",
        description="Dining, travel, parenting, consumer reviews, home design, and societal lifestyle essays.",
        color_accent="#EAB308",
        sample_headlines=[
            "Heres A Cheaper and Smaller Vitamix Blender Thats Totally Worth the Money",
            "How People React When You Tell Them You Are No Longer Drinking Alcohol",
            "Travel Guide: Hidden Destinations and Architectural Highlights in Historic Dublin"
        ]
    ),
    CategoryDetail(
        id="politics",
        name="Politics",
        description="Elections, congress, presidential campaigns, legislative voting, and government policies.",
        color_accent="#EF4444",
        sample_headlines=[
            "Michigan Supreme Court Revives Reproductive Rights Amendment For November Ballot",
            "Jon Stewart Sums Up The Stalled Congressional Legislative Process",
            "Senate Committee Advances Bipartisan Tech Privacy and Antitrust Oversight Legislation"
        ]
    ),
    CategoryDetail(
        id="sports",
        name="Sports",
        description="Athletics championships, NBA, NFL, tennis grand slams, football leagues, and athletic profiles.",
        color_accent="#10B981",
        sample_headlines=[
            "Wisconsin Beats Kentucky in Thrilling Upset to Advance to NCAA Championship Game",
            "Kody Clemens Strikes Out MVP Shohei Ohtani in Memorable Regular Season Duel",
            "Olympic Hammer Thrower Amanda Bingson Celebrates Championship Victory"
        ]
    ),
    CategoryDetail(
        id="technology",
        name="Technology",
        description="Artificial intelligence, software, cybersecurity, smartphone hardware, and internet platforms.",
        color_accent="#06B6D4",
        sample_headlines=[
            "Apple Executive Opposes Federal Backdoor Mandate Protecting Device Cryptography",
            "New Frontier AI Model Demonstrates Mathematical Reasoning and Code Generation",
            "Smartphones and Sleep Quality: Analyzing Screen Time Impact on Rest Architecture"
        ]
    ),
    CategoryDetail(
        id="women-gender",
        name="Women & Gender",
        description="Reproductive rights, gender wage equity, maternal healthcare, and women's leadership representation.",
        color_accent="#D946EF",
        sample_headlines=[
            "Coalition of Female Leaders Launches Initiative to Address Workplace Wage Parity",
            "National Forum Examines Maternal Health Access and Healthcare Disparities",
            "Advocacy Groups Mobilize Support for Gender Equity Protection Legislation"
        ]
    ),
    CategoryDetail(
        id="world-international",
        name="World & International",
        description="Geopolitical foreign affairs, international conflicts, United Nations diplomacy, and global news.",
        color_accent="#F97316",
        sample_headlines=[
            "Pakistan Flooding Deaths Pass One Thousand in Historic Climate Catastrophe",
            "International Inspectors Visit Zaporizhzhia Nuclear Facility to Assess Power Grid Safety",
            "United Nations General Assembly Convenes Global Summit on Refugee Resettlement"
        ]
    ),
]


# ==========================================
# BASELINE v1 REST ENDPOINTS (PRESERVED)
# ==========================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Return system readiness, model loading status, and runtime uptime."""
    return HealthResponse(
        status="healthy" if model_service.is_ready() else "degraded",
        model_loaded=model_service.is_ready(),
        model_name=model_service.model_name,
        model_version=model_service.model_version,
        uptime_seconds=round(time.time() - START_TIME, 2),
        supported_categories=model_service.categories,
        cache_stats=model_service.lru_cache.stats(),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_headline(
    payload: PredictRequest,
    request: Request,
    version: Optional[str] = Query(None, description="Model version ('v1' or 'v2')"),
    db: Session = Depends(get_db)
):
    """
    Classify a news headline. By default provides baseline prediction for backward compatibility;
    supports version='v2' or '13class' for Track 1 production inference.
    """
    if not model_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model pipeline is not ready."
        )

    req_id = getattr(request.state, "request_id", "unknown")
    target_version = version or "v1"

    try:
        if target_version in ["v2", "13class", "track1"]:
            result = model_service.predict_13class(payload.headline)
        else:
            result = model_service.predict_v1(payload.headline)

        # Persist prediction record to SQLite
        record = PredictionRecord(
            headline=result["headline"],
            category=result["category"],
            confidence=result["confidence"],
            probabilities_json=json.dumps(result["probabilities"]),
            latency_ms=result.get("inference_time_ms", 0.0),
            model_version=result.get("model_version", "1.0.0-4class"),
            request_id=req_id,
        )
        db.add(record)
        db.commit()

        return PredictResponse(**result)
    except Exception as e:
        db.rollback()
        logger.error(f"Inference failure [req_id={req_id}]: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal inference failure. Reference error ID: {req_id}"
        )


@app.get("/categories", response_model=CategoriesResponse, tags=["Metadata"])
def get_categories(version: Optional[str] = Query(None)):
    """Return supported topic categories. Defaults to 4 v1 categories; supports version='v2' for all 13."""
    if version in ["v2", "13class", "track1"]:
        return CategoriesResponse(
            total=len(CATEGORIES_METADATA_13),
            categories=CATEGORIES_METADATA_13,
        )
    return CategoriesResponse(
        total=len(CATEGORIES_METADATA_V1),
        categories=CATEGORIES_METADATA_V1,
    )


@app.get("/metrics", tags=["Analytics"])
def get_metrics():
    """Return model evaluation metrics."""
    return model_service.metrics


@app.get("/benchmarks", tags=["Analytics"])
def get_all_benchmarks():
    """Return separate real-world, out-of-domain external, and synthetic benchmark results."""
    manifest_data = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

    manifest_15 = {}
    if MANIFEST_15_PATH.exists():
        with open(MANIFEST_15_PATH, "r", encoding="utf-8") as f:
            manifest_15 = json.load(f)

    return {
        "primary_real_world": model_service.metrics,
        "track1_13class_real_world": model_service.metrics_13class,
        "independent_external": model_service.external_metrics,
        "controlled_synthetic": model_service.synthetic_metrics,
        "dataset_manifest": manifest_data,
        "dataset_manifest_15class": manifest_15,
        "metadata": model_service.metadata,
    }


@app.get("/history", response_model=HistoryResponse, tags=["History"])
def get_history(
    limit: int = Query(50, ge=1, le=200, description="Page size"),
    offset: int = Query(0, ge=0, description="Offset index"),
    category: Optional[str] = Query(None, description="Optional category filter"),
    db: Session = Depends(get_db),
):
    """Retrieve persistent prediction history with filtering, pagination, and stable sorting."""
    query = db.query(PredictionRecord)
    if category:
        query = query.filter(PredictionRecord.category.ilike(category))

    total = query.count()
    records = query.order_by(
        PredictionRecord.created_at.desc(),
        PredictionRecord.id.desc()
    ).offset(offset).limit(limit).all()

    items = [HistoryItem(**r.to_dict()) for r in records]
    return HistoryResponse(total=total, items=items, limit=limit, offset=offset)


@app.delete("/history", tags=["History"])
def clear_history(
    confirm: bool = Query(False, description="Confirmation required"),
    db: Session = Depends(get_db)
):
    """
    Destructive history deletion.
    Important Security Review: For public production deployment, administrative
    authentication (e.g. Bearer token/API key) must be enforced on this endpoint.
    """
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Destructive deletion aborted. You must supply query parameter ?confirm=true to clear history."
        )

    try:
        deleted_count = db.query(PredictionRecord).delete()
        db.commit()
        return {
            "message": "Prediction history cleared successfully.",
            "deleted_records": deleted_count
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute history deletion."
        )


# ==========================================
# TRACK 1 (13-CLASS) REST API V2 ENDPOINTS
# ==========================================

@app.get("/api/v2/health", tags=["Track 1 (13-Class)"])
def v2_health_check():
    """Health check for the 13-class Track 1 model."""
    ready = model_service.pipeline_13class is not None
    return {
        "status": "healthy" if ready else "degraded",
        "model_loaded": ready,
        "model_architecture": "TF-IDF + CalibratedClassifierCV(LinearSVC, cv=5)",
        "model_version": "3.0.0-13class",
        "active_categories_count": len(model_service.categories_13),
        "active_categories": model_service.categories_13,
        "cache_stats": model_service.lru_cache.stats(),
        "uptime_seconds": round(time.time() - START_TIME, 2),
    }


@app.get("/api/v2/categories", tags=["Track 1 (13-Class)"])
def v2_get_categories():
    """Return all 13 active news categories with descriptions, color accents, and sample headlines."""
    return {
        "total_active": len(CATEGORIES_METADATA_13),
        "categories": [c.model_dump() for c in CATEGORIES_METADATA_13],
        "inactive_reserved": [
            {"id": "automobile-transport", "name": "Automobile & Transport", "status": "INACTIVE"},
            {"id": "weather-disasters", "name": "Weather & Natural Disasters", "status": "INACTIVE"}
        ]
    }


@app.post("/api/v2/predict", tags=["Track 1 (13-Class)"])
def v2_predict_headline(
    payload: PredictRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Classify a headline using the 13-category Track 1 model.
    Returns primary category, calibrated probability distribution across all 13 classes,
    token-level feature attributions, and experimental secondary tag analysis.
    """
    req_id = getattr(request.state, "request_id", "unknown")
    try:
        result = model_service.predict_13class(payload.headline)

        record = PredictionRecord(
            headline=result["headline"],
            category=result["category"],
            confidence=result["confidence"],
            probabilities_json=json.dumps(result["probabilities"]),
            latency_ms=result.get("inference_time_ms", 0.0),
            model_version=result.get("model_version", "3.0.0-13class"),
            request_id=req_id,
        )
        db.add(record)
        db.commit()

        return result
    except Exception as e:
        db.rollback()
        logger.error(f"Track 1 inference failure [req_id={req_id}]: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure in 13-class model. Error ref: {req_id}"
        )


@app.get("/api/v2/benchmarks", tags=["Track 1 (13-Class)"])
def v2_get_benchmarks():
    """Return comprehensive evaluation metrics for Track 1 13-class model alongside baselines."""
    return {
        "track1_13class_production": model_service.metrics_13class,
        "baseline_v1_4class": model_service.metrics,
        "pilot_track2_gold_test_meta": {
            "gold_test_file": "backend/data/multilabel_gold_test.csv",
            "records": 50,
            "status": "Experimental Pilot Benchmark"
        }
    }
