"""
Reproducible Agreement Calculation Script.
Computes observed agreement (Po), empirical chance agreement (Pe) from actual marginal distributions,
Cohen's Kappa (kappa), multi-label Jaccard similarity, and binary Hamming agreement
from raw Pass A and Pass B annotation files.
"""

import json
import os
import numpy as np

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

PASS_A_PATH = os.path.join(os.path.dirname(__file__), "pass_a_raw.json")
PASS_B_PATH = os.path.join(os.path.dirname(__file__), "pass_b_raw.json")

def calculate():
    with open(PASS_A_PATH, "r", encoding="utf-8") as f:
        pass_a = json.load(f)
    with open(PASS_B_PATH, "r", encoding="utf-8") as f:
        pass_b = json.load(f)

    assert len(pass_a) == len(pass_b)
    n = len(pass_a)

    cat_to_idx = {c: i for i, c in enumerate(ACTIVE_CATEGORIES)}
    C = np.zeros((len(ACTIVE_CATEGORIES), len(ACTIVE_CATEGORIES)), dtype=int)

    primary_agreements = 0
    jaccards = []
    hamming_matches = 0

    for a, b in zip(pass_a, pass_b):
        assert a["source_record_id"] == b["source_record_id"]
        p_a = a["primary"]
        p_b = b["primary"]
        C[cat_to_idx[p_a], cat_to_idx[p_b]] += 1

        if p_a == p_b:
            primary_agreements += 1

        tags_a = set([p_a] + a.get("secondary", []))
        tags_b = set([p_b] + b.get("secondary", []))
        union = tags_a.union(tags_b)
        jaccard = len(tags_a.intersection(tags_b)) / len(union) if union else 1.0
        jaccards.append(jaccard)

        for cat in ACTIVE_CATEGORIES:
            if (cat in tags_a) == (cat in tags_b):
                hamming_matches += 1

    Po = primary_agreements / n
    row_marginals = C.sum(axis=1) / n
    col_marginals = C.sum(axis=0) / n
    Pe = float(np.sum(row_marginals * col_marginals))
    kappa = (Po - Pe) / (1.0 - Pe)

    mean_jaccard = float(np.mean(jaccards))
    hamming_agreement = hamming_matches / (n * len(ACTIVE_CATEGORIES))
    hamming_loss = 1.0 - hamming_agreement

    print(f"Total Evaluated Records: {n}")
    print(f"Observed Primary Agreement (Po): {Po:.4f} ({primary_agreements}/{n} = {Po*100:.1f}%)")
    print(f"Empirical Expected Agreement (Pe): {Pe:.4f} ({Pe*100:.2f}%)")
    print(f"Exact Cohen's Kappa: {kappa:.4f}")
    print(f"Mean Multi-Label Jaccard Similarity: {mean_jaccard:.4f}")
    print(f"Binary Tag Hamming Agreement: {hamming_agreement:.4f} (Hamming Loss: {hamming_loss:.4f})")
    print("\n--- Empirical Marginal Distributions ---")
    for i, cat in enumerate(ACTIVE_CATEGORIES):
        print(f"  {cat:25}: Annotator A = {row_marginals[i]:.2f} | Annotator B = {col_marginals[i]:.2f} | Prod = {row_marginals[i]*col_marginals[i]:.4f}")

if __name__ == "__main__":
    calculate()
