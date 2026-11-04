"""
SQLite Database setup using SQLAlchemy for persistent prediction history.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

DB_DIR = Path(__file__).resolve().parent.parent / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "predictions.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    headline = Column(Text, nullable=False)
    category = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities_json = Column(Text, nullable=False)
    latency_ms = Column(Float, nullable=True, default=0.0)
    model_version = Column(String(50), nullable=True, default="3.0.0-13class")
    request_id = Column(String(64), nullable=True, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "headline": self.headline,
            "category": self.category,
            "confidence": round(self.confidence, 4),
            "probabilities": json.loads(self.probabilities_json),
            "latency_ms": round(self.latency_ms or 0.0, 2),
            "model_version": self.model_version or "3.0.0-13class",
            "request_id": self.request_id or "",
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }


def init_db():
    Base.metadata.create_all(bind=engine)
    # Lightweight schema migration for existing SQLite databases
    with engine.connect() as conn:
        for col_name, col_type in [
            ("latency_ms", "FLOAT DEFAULT 0.0"),
            ("model_version", "VARCHAR(50) DEFAULT '3.0.0-13class'"),
            ("request_id", "VARCHAR(64) DEFAULT ''"),
        ]:
            try:
                conn.execute(text(f"ALTER TABLE predictions ADD COLUMN {col_name} {col_type}"))
                conn.commit()
            except Exception:
                pass


# Initialize tables on load
init_db()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
