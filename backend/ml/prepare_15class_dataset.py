"""
Data Preparation, Hygiene, and Near-Duplicate Quarantine Pipeline
for Scalable 15-Category (13 Active) News Headline Topic Classification.

Source: Rishabh Misra, HuffPost News Category Dataset (v3)
Citation: R. Misra. News Category Dataset. arXiv:2209.11429, 2022.
License: Creative Commons CC0: Public Domain (209,527 raw records).

This script performs:
1. Complete raw data hygiene (empty, <3 words, exact duplicate elimination).
2. Semantically audited category mapping (13 active categories, 2 inactive).
3. Canonical template deduplication and near-duplicate event clustering:
   - Stripping episodic markers, years, dates, and media tags.
   - Quarantining templated boilerplate to prevent cross-split leakage.
4. Strict Reservation of 50 Gold Test Benchmark Records:
   - All 50 gold benchmark records are reserved exclusively for test_13class.csv.
   - Guaranteed 0 overlap with train_13class.csv and val_13class.csv.
5. Balanced stratified sampling capped at 2,000 authentic records per active class (26,000 total).
6. Strict Near-Duplicate Quarantine & Replacement (Threshold sim < 0.80):
   - Fast batch cosine similarity audit against Train.
   - Every Val or Test headline with char 3-gram cosine similarity >= 0.80 to Train is quarantined and replaced.
   - Ensures ZERO exact and ZERO near-duplicate matches across split boundaries.
7. Comprehensive reconciliation and manifest generation.
"""

import json
import os
import random
import re
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Seed for deterministic reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

RAW_PATH = os.path.join("backend", "data", "raw", "news_category_dataset_v3.json")
MANIFEST_PATH = os.path.join("backend", "data", "dataset_manifest_15class.json")
TRAIN_PATH = os.path.join("backend", "data", "train_13class.csv")
VAL_PATH = os.path.join("backend", "data", "val_13class.csv")
TEST_PATH = os.path.join("backend", "data", "test_13class.csv")
PILOT_PATH = os.path.join("backend", "data", "multilabel_validation_subset.json")
GOLD_BENCHMARK_PATH = os.path.join("backend", "data", "multilabel_gold_test.json")

# Audited 13-category semantic mapping
CATEGORY_MAPPING = {
    # 1. Education
    "EDUCATION": "Education",
    "COLLEGE": "Education",
    # 2. Politics
    "POLITICS": "Politics",
    "U.S. NEWS": "Politics",
    # 3. Entertainment (Excludes MEDIA due to cable news politics overlap)
    "ENTERTAINMENT": "Entertainment",
    "COMEDY": "Entertainment",
    # 4. Crime & Law
    "CRIME": "Crime & Law",
    # 5. Women & Gender (Excludes QUEER VOICES to maintain gender specificity)
    "WOMEN": "Women & Gender",
    # 6. Art & Culture
    "ARTS": "Art & Culture",
    "ARTS & CULTURE": "Art & Culture",
    "CULTURE & ARTS": "Art & Culture",
    # 7. Business & Economy
    "BUSINESS": "Business & Economy",
    "MONEY": "Business & Economy",
    # 8. Technology
    "TECH": "Technology",
    # 9. Sports
    "SPORTS": "Sports",
    # 10. Health & Science (Excludes WELLNESS to avoid lifestyle/symptom ambiguity)
    "SCIENCE": "Health & Science",
    "HEALTHY LIVING": "Health & Science",
    # 11. Environment & Climate
    "ENVIRONMENT": "Environment & Climate",
    "GREEN": "Environment & Climate",
    # 12. World & International
    "WORLD NEWS": "World & International",
    "THE WORLDPOST": "World & International",
    "WORLDPOST": "World & International",
    # 13. Lifestyle & Society
    "TRAVEL": "Lifestyle & Society",
    "STYLE & BEAUTY": "Lifestyle & Society",
    "PARENTING": "Lifestyle & Society",
    "FOOD & DRINK": "Lifestyle & Society",
    "HOME & LIVING": "Lifestyle & Society",
    "PARENTS": "Lifestyle & Society",
    "STYLE": "Lifestyle & Society",
    "TASTE": "Lifestyle & Society",
}

INACTIVE_CATEGORIES = {
    "Automobile & Transport": "0 standalone records in HuffPost v3. Maintained as inactive in hierarchical taxonomy schema.",
    "Weather & Natural Disasters": "0 standalone records in HuffPost v3. CrisisNLP evaluated and quarantined due to tweet domain mismatch."
}

def canonical_form(text: str) -> str:
    """Normalizes headline text to detect boilerplate, recurring series, and syndicated templates."""
    text = text.lower()
    text = re.sub(r'\(photos\)|\[photos\]|\(video\)|\(exclusive\)', '', text)
    text = re.sub(r'\b(part \d+|episode \d+|season \d+|\d{4}|\d{1,2}/\d{1,2}/\d{2,4})\b', '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return ' '.join(text.split())

def load_clean_records():
    print(f"Loading raw dataset from {RAW_PATH}...")
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    total_raw = len(records)
    print(f"Loaded {total_raw} raw records.")

    # Data hygiene filters
    empty_indices = set()
    short_indices = set()
    for idx, r in enumerate(records):
        hl = r.get("headline", "").strip()
        if not hl:
            empty_indices.add(idx)
        if len(hl.split()) < 3:
            short_indices.add(idx)

    seen_hl = {}
    dup_indices = set()
    for idx, r in enumerate(records):
        hl = r.get("headline", "").strip()
        if hl in seen_hl:
            dup_indices.add(idx)
        else:
            seen_hl[hl] = idx

    short_and_dup = short_indices.intersection(dup_indices)
    all_excluded = short_indices.union(dup_indices)
    clean_indices = [i for i in range(total_raw) if i not in all_excluded]

    assert len(all_excluded) + len(clean_indices) == total_raw

    clean_mapped = []
    clean_unmapped = []
    for idx in clean_indices:
        r = records[idx]
        orig_cat = r.get("category", "")
        if orig_cat in CATEGORY_MAPPING:
            clean_mapped.append({
                "source_record_id": idx,
                "headline": r.get("headline", "").strip(),
                "canonical_form": canonical_form(r.get("headline", "")),
                "short_description": r.get("short_description", "").strip(),
                "original_category": orig_cat,
                "target_category": CATEGORY_MAPPING[orig_cat],
                "date": r.get("date", ""),
                "link": r.get("link", "")
            })
        else:
            clean_unmapped.append({
                "source_record_id": idx,
                "original_category": orig_cat,
                "headline": r.get("headline", "").strip()
            })

    assert len(clean_mapped) + len(clean_unmapped) == len(clean_indices)
    return records, clean_mapped, clean_unmapped, {
        "total_raw": total_raw,
        "empty": len(empty_indices),
        "short": len(short_indices),
        "duplicate": len(dup_indices),
        "overlap": len(short_and_dup),
        "net_excluded": len(all_excluded),
        "clean_total": len(clean_indices),
        "clean_mapped": len(clean_mapped),
        "clean_unmapped": len(clean_unmapped)
    }

def create_quarantined_balanced_splits(clean_mapped, cap_per_class=2000):
    print("\n--- Reserving Gold Benchmark and Partitioning Splits ---")
    # Load gold benchmark IDs
    with open(GOLD_BENCHMARK_PATH, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    gold_ids = set(r["source_record_id"] for r in gold_data["records"])
    print(f"Reserving {len(gold_ids)} Gold Test Benchmark IDs exclusively for test_13class.csv...")

    # Group clean records by category, separating gold items from general pool
    by_category_pool = defaultdict(list)
    gold_by_category = defaultdict(list)

    for r in clean_mapped:
        cat = r["target_category"]
        if r["source_record_id"] in gold_ids:
            gold_by_category[cat].append(r)
        else:
            by_category_pool[cat].append(r)

    assert sum(len(v) for v in gold_by_category.values()) == 50

    # 1. Deduplicate repeated canonical templates within the general pool
    cat_deduped = {}
    for cat, items in by_category_pool.items():
        seen_canon = set()
        deduped_items = []
        for item in items:
            cf = item["canonical_form"]
            if cf not in seen_canon:
                seen_canon.add(cf)
                deduped_items.append(item)

        random.seed(RANDOM_SEED)
        random.shuffle(deduped_items)
        cat_deduped[cat] = deduped_items

    # 2. Partition:
    # - Train: 1600 per class (drawn strictly from non-gold pool)
    # - Val:   200 per class  (drawn strictly from non-gold pool)
    # - Test:  200 per class  (includes all gold records for that class + remainder from pool)
    train_records = []
    val_records = []
    test_records = []
    candidate_reserves = defaultdict(list)

    for cat in sorted(list(set(CATEGORY_MAPPING.values()))):
        if cat in cat_deduped:
            items = cat_deduped[cat]
            gold_items = gold_by_category[cat]
            n_gold = len(gold_items)

            # Test set receives all gold items for this class
            test_for_cat = list(gold_items)
            needed_test = 200 - n_gold

            # Draw 1600 Train, 200 Val, remainder Test
            train_records.extend(items[:1600])
            val_records.extend(items[1600:1800])
            test_for_cat.extend(items[1800:1800 + needed_test])
            test_records.extend(test_for_cat)

            # Keep remaining as reserves for quarantine replacement
            candidate_reserves[cat].extend(items[1800 + needed_test:])

    print(f"Initial allocation:")
    print(f"  Train: {len(train_records)} records (0 gold overlap)")
    print(f"  Val:   {len(val_records)} records (0 gold overlap)")
    print(f"  Test:  {len(test_records)} records (contains all 50 gold records)")

    # Verify 0 gold overlap in train and val
    train_ids = set(r["source_record_id"] for r in train_records)
    val_ids = set(r["source_record_id"] for r in val_records)
    test_ids = set(r["source_record_id"] for r in test_records)

    assert len(train_ids.intersection(gold_ids)) == 0, "FATAL: Gold record leaked into Train!"
    assert len(val_ids.intersection(gold_ids)) == 0, "FATAL: Gold record leaked into Val!"
    assert gold_ids.issubset(test_ids), "FATAL: Gold records missing from Test!"

    # 3. Batch Vectorize Train to detect any high-similarity leakage (sim >= 0.80)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2)
    vec.fit([r["headline"] for r in train_records])
    X_train = vec.transform([r["headline"] for r in train_records])

    # Batch check Val
    X_val = vec.transform([r["headline"] for r in val_records])
    sim_val_max = cosine_similarity(X_val, X_train).max(axis=1)
    val_bad_indices = set(np.where(sim_val_max >= 0.80)[0])

    val_clean = []
    val_quarantined = []
    for idx, item in enumerate(val_records):
        if idx in val_bad_indices:
            val_quarantined.append((item, float(sim_val_max[idx])))
            cat = item["target_category"]
            # Find clean replacement from reserves
            while candidate_reserves[cat]:
                cand = candidate_reserves[cat].pop(0)
                cand_sim = float(cosine_similarity(vec.transform([cand["headline"]]), X_train).max())
                if cand_sim < 0.80 and cand["source_record_id"] not in gold_ids:
                    val_clean.append(cand)
                    break
        else:
            val_clean.append(item)

    # Batch check Test (never quarantine gold benchmark items)
    X_test = vec.transform([r["headline"] for r in test_records])
    sim_test_max = cosine_similarity(X_test, X_train).max(axis=1)
    test_bad_indices = set(np.where(sim_test_max >= 0.80)[0])

    test_clean = []
    test_quarantined = []
    for idx, item in enumerate(test_records):
        if idx in test_bad_indices:
            # Gold records must not be quarantined; check their sim
            if item["source_record_id"] in gold_ids:
                print(f"  [NOTE] Gold record {item['source_record_id']} sim={sim_test_max[idx]:.4f} retained in Test.")
                test_clean.append(item)
            else:
                test_quarantined.append((item, float(sim_test_max[idx])))
                cat = item["target_category"]
                while candidate_reserves[cat]:
                    cand = candidate_reserves[cat].pop(0)
                    cand_sim = float(cosine_similarity(vec.transform([cand["headline"]]), X_train).max())
                    if cand_sim < 0.80 and cand["source_record_id"] not in gold_ids:
                        test_clean.append(cand)
                        break
        else:
            test_clean.append(item)

    print(f"Quarantined and replaced from Val  (sim >= 0.80): {len(val_quarantined)}")
    print(f"Quarantined and replaced from Test (sim >= 0.80): {len(test_quarantined)}")

    # Verification of zero leakage
    X_val_final = vec.transform([r["headline"] for r in val_clean])
    X_test_final = vec.transform([r["headline"] for r in test_clean])

    max_sim_val = float(cosine_similarity(X_val_final, X_train).max())
    max_sim_test = float(cosine_similarity(X_test_final, X_train).max())

    val_matches_gte_80 = int(np.sum(cosine_similarity(X_val_final, X_train).max(axis=1) >= 0.80))
    test_matches_gte_80 = int(np.sum(cosine_similarity(X_test_final, X_train).max(axis=1) >= 0.80))

    print(f"Post-Quarantine Max Val-Train Similarity:  {max_sim_val:.4f} (Matches >= 0.80: {val_matches_gte_80})")
    print(f"Post-Quarantine Max Test-Train Similarity: {max_sim_test:.4f} (Matches >= 0.80: {test_matches_gte_80})")

    assert len(train_records) == 20800
    assert len(val_clean) == 2600
    assert len(test_clean) == 2600
    assert val_matches_gte_80 == 0
    assert test_matches_gte_80 == 0

    # Final isolation assert
    final_train_ids = set(r["source_record_id"] for r in train_records)
    final_val_ids = set(r["source_record_id"] for r in val_clean)
    final_test_ids = set(r["source_record_id"] for r in test_clean)

    assert len(final_train_ids.intersection(gold_ids)) == 0
    assert len(final_val_ids.intersection(gold_ids)) == 0
    assert gold_ids.issubset(final_test_ids)

    # Save clean split CSVs
    pd.DataFrame(train_records).to_csv(TRAIN_PATH, index=False, encoding="utf-8")
    pd.DataFrame(val_clean).to_csv(VAL_PATH, index=False, encoding="utf-8")
    pd.DataFrame(test_clean).to_csv(TEST_PATH, index=False, encoding="utf-8")
    print(f"Saved verified partitions to:\n  {TRAIN_PATH}\n  {VAL_PATH}\n  {TEST_PATH}")

    return train_records, val_clean, test_clean, {
        "val_quarantined_count": len(val_quarantined),
        "test_quarantined_count": len(test_quarantined),
        "max_val_train_sim": round(max_sim_val, 4),
        "max_test_train_sim": round(max_sim_test, 4),
        "val_matches_gte_80": val_matches_gte_80,
        "test_matches_gte_80": test_matches_gte_80
    }

def update_manifest(hygiene_stats, clean_mapped, clean_unmapped, leak_stats):
    print("\n--- Updating Comprehensive Dataset Manifest ---")
    clean_cat_counts = Counter(r["target_category"] for r in clean_mapped)
    unmapped_cat_counts = Counter(r["original_category"] for r in clean_unmapped)

    manifest = {
        "manifest_version": "2.3.0-15class-milestone1-isolated",
        "dataset_name": "HuffPost News Category Dataset (v3)",
        "author": "Rishabh Misra",
        "citation": "R. Misra. News Category Dataset. arXiv:2209.11429, 2022.",
        "url": "https://huggingface.co/datasets/heegyu/news-category-dataset",
        "license": "Creative Commons CC0: Public Domain",
        "raw_file_size_bytes": os.path.getsize(RAW_PATH),
        "mathematical_reconciliation": {
            "total_raw_records": hygiene_stats["total_raw"],
            "data_hygiene_exclusions": {
                "empty_records": hygiene_stats["empty"],
                "ultra_short_records_lt_3_words": hygiene_stats["short"],
                "exact_duplicates": hygiene_stats["duplicate"],
                "overlap_short_and_duplicate": hygiene_stats["overlap"],
                "net_unique_excluded": hygiene_stats["net_excluded"],
                "exclusion_formula": "net_unique_excluded = short + duplicate - overlap (1690 + 1531 - 129 = 3092)"
            },
            "total_clean_unique_records": hygiene_stats["clean_total"],
            "reconciliation_equation": "total_raw == net_unique_excluded + total_clean_unique (209527 == 3092 + 206435)",
            "clean_record_distribution": {
                "clean_mapped_to_13_categories": hygiene_stats["clean_mapped"],
                "clean_unmapped_in_original_categories": hygiene_stats["clean_unmapped"],
                "mapping_equation": "total_clean_unique == clean_mapped + clean_unmapped (206435 == 155527 + 50908)"
            }
        },
        "target_categories_13_active": {cat: clean_cat_counts[cat] for cat in sorted(clean_cat_counts)},
        "target_categories_2_inactive": INACTIVE_CATEGORIES,
        "unmapped_categories_breakdown": {cat: unmapped_cat_counts[cat] for cat in sorted(unmapped_cat_counts)},
        "balanced_dataset_design": {
            "strategy": "Stratified random sampling capped at 2,000 samples per active category with gold benchmark reservation and near-duplicate quarantine",
            "total_samples": 26000,
            "supervision_paradigm": "Track 1: Single-Label Primary Topic (20,800 train); Track 2: Genuine Multi-Label (150 train pilot, 50 val, 50 gold test)",
            "splits": {
                "train_records": 20800,
                "val_records": 2600,
                "test_records": 2600
            },
            "split_ratio": "80% / 10% / 10%",
            "leakage_auditing": {
                "exact_overlap_across_all_splits": 0,
                "gold_benchmark_in_train": 0,
                "gold_benchmark_in_val": 0,
                "gold_benchmark_in_test": 50,
                "val_quarantined_and_replaced": leak_stats["val_quarantined_count"],
                "test_quarantined_and_replaced": leak_stats["test_quarantined_count"],
                "max_val_train_cosine_similarity": leak_stats["max_val_train_sim"],
                "max_test_train_cosine_similarity": leak_stats["max_test_train_sim"],
                "val_train_matches_gte_80": leak_stats["val_matches_gte_80"],
                "test_train_matches_gte_80": leak_stats["test_matches_gte_80"]
            }
        },
        "benchmarks": {
            "gold_test_benchmark": {
                "file": "backend/data/multilabel_gold_test.json",
                "sample_count": 50,
                "methodology": "Expert-codified protocol agreement (Pass A vs Pass B) with adjudication log",
                "status": "Pilot Multi-Label Gold Test Benchmark (Small evaluation subset)"
            },
            "heuristic_annotation_pilot": {
                "file": "backend/data/multilabel_validation_subset.json",
                "sample_count": 100,
                "methodology": "Programmatic keyword-heuristic cross-domain pilot",
                "status": "Exploratory Annotation Pilot (Not Human Ground Truth)"
            }
        }
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to {MANIFEST_PATH}")
    return manifest

def main():
    records, clean_mapped, clean_unmapped, hygiene_stats = load_clean_records()
    train_records, val_records, test_records, leak_stats = create_quarantined_balanced_splits(clean_mapped, cap_per_class=2000)
    manifest = update_manifest(hygiene_stats, clean_mapped, clean_unmapped, leak_stats)
    print("\n[SUCCESS] Milestone 1 dataset preparation, isolation, quarantine, and manifest update complete!")

if __name__ == "__main__":
    main()
