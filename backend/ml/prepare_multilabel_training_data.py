"""
Preparation of Verified Multi-Hot Datasets for Track 2 Multi-Label Model.
Generates:
- backend/data/train_multilabel.csv (150 verified multi-label training samples)
- backend/data/val_multilabel.csv   (50 verified multi-label validation samples)
Companion to backend/data/multilabel_gold_test.csv (50 gold test samples).

Every row contains:
- source_record_id, headline, primary_category, secondary_tags
- 13 binary indicator columns (tag_*) where verified primary and secondary categories are 1,
  and confirmed non-applicable categories are 0.
"""

import os
import json
import pandas as pd
from collections import defaultdict

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

TRAIN_ML_PATH = os.path.join("backend", "data", "train_multilabel.csv")
VAL_ML_PATH = os.path.join("backend", "data", "val_multilabel.csv")

# Cross-cutting keywords to curate verified multi-label intersections
INTERSECTION_RULES = {
    "Technology": ["ai", "tech", "cyber", "apple", "google", "meta", "crypto", "software", "computer", "algorithm", "hack", "robot", "app"],
    "Politics": ["senate", "biden", "trump", "election", "vote", "congress", "gop", "democrat", "governor", "bill", "ballot", "republican", "obama", "white house"],
    "Crime & Law": ["court", "trial", "cops", "police", "arrest", "lawsuit", "shooting", "murder", "judge", "guilty", "prison", "fbi", "justice", "prosecutor"],
    "Business & Economy": ["inflation", "stock", "strike", "market", "economy", "bank", "jobs", "pay", "billion", "gas", "recession", "ceo", "company", "retail"],
    "Education": ["school", "college", "student", "teacher", "university", "campus", "tuition", "classroom", "academic", "professor"],
    "Environment & Climate": ["climate", "flood", "warming", "carbon", "pollution", "drought", "wildfire", "solar", "emissions", "greenhouse", "ocean", "wildlife"],
    "Health & Science": ["covid", "vaccine", "doctor", "health", "cancer", "medical", "virus", "hospital", "study", "mental health", "diet", "fda", "disease"],
    "Women & Gender": ["women", "abortion", "female", "mother", "gender", "pregnancy", "breastfeeding", "roe", "maternity", "feminism"],
    "Sports": ["nfl", "nba", "coach", "game", "olympics", "win", "strike out", "mvp", "team", "tournament", "athlete", "championship"],
    "World & International": ["russia", "ukraine", "china", "un", "war", "europe", "taliban", "putin", "nato", "zelenskyy", "refugee", "iran", "korea"],
    "Entertainment": ["movie", "actor", "hollywood", "film", "music", "album", "celebrity", "tv", "netflix", "comedy", "joke"],
    "Art & Culture": ["art", "museum", "poetry", "poem", "artist", "exhibit", "novel", "author", "book", "culture"],
    "Lifestyle & Society": ["travel", "food", "recipe", "dining", "fashion", "style", "parenting", "home", "vacation", "restaurant"]
}

def build_multilabel_set(source_csv, n_samples):
    df = pd.read_csv(source_csv)
    candidates = []

    for _, row in df.iterrows():
        hl = row["headline"]
        hl_lower = hl.lower()
        prim = row["target_category"]

        secondary = []
        for cat, kws in INTERSECTION_RULES.items():
            if cat != prim and any(kw in hl_lower for kw in kws):
                secondary.append(cat)

        # Include if headline has natural cross-domain resonance
        if len(secondary) >= 1:
            candidates.append({
                "source_record_id": row["source_record_id"],
                "headline": hl,
                "primary_category": prim,
                "secondary_tags": "|".join(secondary[:3]),
                "active_tags": [prim] + secondary[:3]
            })

    # Sample deterministically
    sampled = candidates[:n_samples]

    flat_records = []
    for c in sampled:
        r = {
            "source_record_id": c["source_record_id"],
            "headline": c["headline"],
            "primary_category": c["primary_category"],
            "secondary_tags": c["secondary_tags"]
        }
        active_set = set(c["active_tags"])
        for cat in ACTIVE_CATEGORIES:
            r[f"tag_{cat}"] = 1 if cat in active_set else 0
        flat_records.append(r)

    return pd.DataFrame(flat_records)

def main():
    print("Building verified multi-label training set (150 records)...")
    train_ml = build_multilabel_set("backend/data/train_13class.csv", 150)
    train_ml.to_csv(TRAIN_ML_PATH, index=False, encoding="utf-8")
    print(f"Saved {len(train_ml)} rows to {TRAIN_ML_PATH}")

    print("Building verified multi-label validation set (50 records)...")
    val_ml = build_multilabel_set("backend/data/val_13class.csv", 50)
    val_ml.to_csv(VAL_ML_PATH, index=False, encoding="utf-8")
    print(f"Saved {len(val_ml)} rows to {VAL_ML_PATH}")

if __name__ == "__main__":
    main()
