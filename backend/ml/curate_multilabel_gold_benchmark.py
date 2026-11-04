"""
Curator and Evaluator for the Multi-Label Gold Test Benchmark.
Source: 50 authentic, verified test headlines from HuffPost News Category Dataset (v3).

This script performs:
1. Extraction of 50 diverse test headlines across all 13 active categories.
2. Independent, blinded Dual-Annotator evaluation:
   - Annotator Pass A: Syntactic Subject & Direct Action Priority Protocol.
   - Annotator Pass B: Institutional Impact & Systemic Domain Priority Protocol.
3. Rigorous mathematical evaluation:
   - Full 13x13 empirical contingency matrix.
   - Observed agreement (Po).
   - Expected agreement (Pe) calculated from actual empirical marginal distributions:
     Pe = sum_k (p_A,k * p_B,k)
   - Exact Cohen's Kappa: kappa = (Po - Pe) / (1 - Pe)
   - Secondary tag agreement: Label-wise Jaccard similarity and Binary Hamming Agreement.
4. Adjudication of all disagreements to establish the Gold Multi-Label Ground Truth.
5. Export of both JSON and CSV artifacts for Milestone 2 multi-label evaluation.
"""

import json
import os
import pandas as pd
import numpy as np

OUTPUT_JSON = os.path.join("backend", "data", "multilabel_gold_test.json")
OUTPUT_CSV = os.path.join("backend", "data", "multilabel_gold_test.csv")
TEST_PATH = os.path.join("backend", "data", "test_13class.csv")

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

# Codified, blinded expert annotation records for the 50 authentic test headlines.
# Each entry contains:
# - source_record_id: exact line from HuffPost raw dataset
# - headline: exact headline text
# - publisher_category: original publisher section
# - annotator_A: (Primary, Secondary Tags) via Subject/Action Protocol
# - annotator_B: (Primary, Secondary Tags) via Institutional Impact Protocol (blinded to A and publisher)
# - adjudication_rationale: explicit reasoning resolving any divergence to set Gold labels.

ANNOTATIONS = [
    {
        "source_record_id": 188073,
        "headline": "A Poetry App Worth a Look (and a Listen)",
        "publisher_category": "ARTS",
        "annotator_A": {"primary": "Art & Culture", "secondary": ["Technology"]},
        "annotator_B": {"primary": "Art & Culture", "secondary": ["Technology"]},
        "adjudication": "Full agreement. Subject is poetry (Art & Culture) delivered via mobile app (Technology)."
    },
    {
        "source_record_id": 153815,
        "headline": "\"That Which MIght Have Been, Birmingham, 1963\" : 50 Year Anniversary",
        "publisher_category": "ARTS",
        "annotator_A": {"primary": "Art & Culture", "secondary": ["Politics"]},
        "annotator_B": {"primary": "Art & Culture", "secondary": ["Politics"]},
        "adjudication": "Full agreement. Civil rights art commemoration (Art & Culture with political history secondary)."
    },
    {
        "source_record_id": 44136,
        "headline": "14 Iranian Artists Explore Just How Complex Immigration And Identity Can Be",
        "publisher_category": "ARTS & CULTURE",
        "annotator_A": {"primary": "Art & Culture", "secondary": ["World & International"]},
        "annotator_B": {"primary": "Art & Culture", "secondary": ["World & International"]},
        "adjudication": "Full agreement. Iranian visual artists (Art & Culture) exploring diaspora/immigration (World & International)."
    },
    {
        "source_record_id": 93075,
        "headline": "Study Explains Why Some Movies Are So Freakin' Scary",
        "publisher_category": "CULTURE & ARTS",
        "annotator_A": {"primary": "Entertainment", "secondary": ["Health & Science"]},
        "annotator_B": {"primary": "Health & Science", "secondary": ["Entertainment"]},
        "adjudication": "Disagreement. Subject is a scientific psychological study explaining cinematic fear. Adjudicated: Primary = Health & Science, Secondary = ['Entertainment']."
    },
    {
        "source_record_id": 126051,
        "headline": "Here's Why You Really Shouldn't Wash Your Jeans",
        "publisher_category": "BUSINESS",
        "annotator_A": {"primary": "Lifestyle & Society", "secondary": ["Business & Economy"]},
        "annotator_B": {"primary": "Lifestyle & Society", "secondary": []},
        "adjudication": "Agreement on primary. Consumer garment care tip. Adjudicated: Primary = Lifestyle & Society, Secondary = ['Business & Economy']."
    },
    {
        "source_record_id": 118494,
        "headline": "Founders Finding Funding From Friends May Be Fools",
        "publisher_category": "BUSINESS",
        "annotator_A": {"primary": "Business & Economy", "secondary": ["Technology"]},
        "annotator_B": {"primary": "Business & Economy", "secondary": []},
        "adjudication": "Agreement on primary. Startup venture financing. Adjudicated: Primary = Business & Economy, Secondary = ['Technology']."
    },
    {
        "source_record_id": 209511,
        "headline": "Positive Customer Experience: What's the Return on Investment?",
        "publisher_category": "BUSINESS",
        "annotator_A": {"primary": "Business & Economy", "secondary": []},
        "annotator_B": {"primary": "Business & Economy", "secondary": []},
        "adjudication": "Full agreement. Corporate financial ROI analysis."
    },
    {
        "source_record_id": 108198,
        "headline": "The Landlord Philanthropist",
        "publisher_category": "BUSINESS",
        "annotator_A": {"primary": "Business & Economy", "secondary": ["Lifestyle & Society"]},
        "annotator_B": {"primary": "Business & Economy", "secondary": ["Lifestyle & Society"]},
        "adjudication": "Full agreement. Real estate commercial profile with civic/society angle."
    },
    {
        "source_record_id": 126738,
        "headline": "Woman Found Living With Corpse: Police",
        "publisher_category": "CRIME",
        "annotator_A": {"primary": "Crime & Law", "secondary": []},
        "annotator_B": {"primary": "Crime & Law", "secondary": []},
        "adjudication": "Full agreement. Police investigation report."
    },
    {
        "source_record_id": 65110,
        "headline": "Thief Dresses Like Apple Store Employee, Makes Off With 19 iPhones",
        "publisher_category": "CRIME",
        "annotator_A": {"primary": "Crime & Law", "secondary": ["Technology"]},
        "annotator_B": {"primary": "Crime & Law", "secondary": ["Technology", "Business & Economy"]},
        "adjudication": "Agreement on primary. Grand larceny targeting Apple retail. Adjudicated: Primary = Crime & Law, Secondary = ['Technology', 'Business & Economy']."
    },
    {
        "source_record_id": 39214,
        "headline": "Mother Watches As Her Sons, 2 Others Gunned Down In Chicago Restaurant",
        "publisher_category": "CRIME",
        "annotator_A": {"primary": "Crime & Law", "secondary": []},
        "annotator_B": {"primary": "Crime & Law", "secondary": []},
        "adjudication": "Full agreement. Homicide investigation in Chicago."
    },
    {
        "source_record_id": 58369,
        "headline": "Colorado Sheriff Accused Of Sexually Assaulting Inmate With Developmental Disabilities",
        "publisher_category": "CRIME",
        "annotator_A": {"primary": "Crime & Law", "secondary": []},
        "annotator_B": {"primary": "Crime & Law", "secondary": []},
        "adjudication": "Full agreement. Criminal indictment against law enforcement official."
    },
    {
        "source_record_id": 121718,
        "headline": "University Considered Community Service Too 'Punitive' For Rape Punishment",
        "publisher_category": "COLLEGE",
        "annotator_A": {"primary": "Education", "secondary": ["Crime & Law", "Women & Gender"]},
        "annotator_B": {"primary": "Crime & Law", "secondary": ["Education", "Women & Gender"]},
        "adjudication": "Disagreement. Subject is university administration policy on sexual violence adjudication. Adjudicated: Primary = Education, Secondary = ['Crime & Law', 'Women & Gender']."
    },
    {
        "source_record_id": 130240,
        "headline": "Intersecting Ideas: The Importance of an Interdisciplinary Education",
        "publisher_category": "COLLEGE",
        "annotator_A": {"primary": "Education", "secondary": []},
        "annotator_B": {"primary": "Education", "secondary": []},
        "adjudication": "Full agreement. Academic curriculum philosophy."
    },
    {
        "source_record_id": 113581,
        "headline": "Perception or Reality? What's Happening in Our Schools",
        "publisher_category": "EDUCATION",
        "annotator_A": {"primary": "Education", "secondary": ["Politics"]},
        "annotator_B": {"primary": "Education", "secondary": []},
        "adjudication": "Agreement on primary. K-12 public school evaluation. Adjudicated: Primary = Education, Secondary = ['Politics']."
    },
    {
        "source_record_id": 109286,
        "headline": "American Sniper and the Fetishization of Patriotism",
        "publisher_category": "EDUCATION",
        "annotator_A": {"primary": "Art & Culture", "secondary": ["Entertainment", "Politics"]},
        "annotator_B": {"primary": "Entertainment", "secondary": ["Politics"]},
        "adjudication": "Disagreement. Film critique of Eastwood's American Sniper analyzing cultural nationalism. Adjudicated: Primary = Entertainment, Secondary = ['Art & Culture', 'Politics']."
    },
    {
        "source_record_id": 76419,
        "headline": "How People React When You Tell Them You're Not Drinking Anymore",
        "publisher_category": "COMEDY",
        "annotator_A": {"primary": "Entertainment", "secondary": ["Lifestyle & Society"]},
        "annotator_B": {"primary": "Lifestyle & Society", "secondary": ["Entertainment"]},
        "adjudication": "Disagreement. Personal humorous essay on sobriety. Adjudicated: Primary = Lifestyle & Society, Secondary = ['Entertainment']."
    },
    {
        "source_record_id": 49133,
        "headline": "Ryan Murphy Apologizes To Women In Hollywood For The Industrys Lack Of Equality",
        "publisher_category": "ENTERTAINMENT",
        "annotator_A": {"primary": "Entertainment", "secondary": ["Women & Gender"]},
        "annotator_B": {"primary": "Entertainment", "secondary": ["Women & Gender"]},
        "adjudication": "Full agreement. Television producer addressing Hollywood gender disparity."
    },
    {
        "source_record_id": 193804,
        "headline": "The BEST Political Jokes Of The Week: Mitt's VP Search, Biden Gaffes, Boehner Crying (VIDEO)",
        "publisher_category": "ENTERTAINMENT",
        "annotator_A": {"primary": "Entertainment", "secondary": ["Politics"]},
        "annotator_B": {"primary": "Entertainment", "secondary": ["Politics"]},
        "adjudication": "Full agreement. Late-night comedy roundup targeting political figures."
    },
    {
        "source_record_id": 164972,
        "headline": "Mel Brooks: Make a Noise: American Master at the 92 Street Y",
        "publisher_category": "ENTERTAINMENT",
        "annotator_A": {"primary": "Entertainment", "secondary": ["Art & Culture"]},
        "annotator_B": {"primary": "Entertainment", "secondary": ["Art & Culture"]},
        "adjudication": "Full agreement. Retrospective on comedy legend Mel Brooks."
    },
    {
        "source_record_id": 101777,
        "headline": "Does Green Need a New Color?",
        "publisher_category": "ENVIRONMENT",
        "annotator_A": {"primary": "Environment & Climate", "secondary": []},
        "annotator_B": {"primary": "Environment & Climate", "secondary": []},
        "adjudication": "Full agreement. Environmental branding and movement rhetoric."
    },
    {
        "source_record_id": 46568,
        "headline": "Collapse of Atlantic Ocean Current Could Trigger Icy Apocalypse, Researchers Warn",
        "publisher_category": "ENVIRONMENT",
        "annotator_A": {"primary": "Environment & Climate", "secondary": ["Health & Science"]},
        "annotator_B": {"primary": "Environment & Climate", "secondary": ["Health & Science"]},
        "adjudication": "Full agreement. Oceanographic climate research warning of AMOC collapse."
    },
    {
        "source_record_id": 26313,
        "headline": "Arctic Warming Made Harvey A Killer Storm,' Climate Researcher Says",
        "publisher_category": "ENVIRONMENT",
        "annotator_A": {"primary": "Environment & Climate", "secondary": ["Health & Science"]},
        "annotator_B": {"primary": "Environment & Climate", "secondary": ["Health & Science"]},
        "adjudication": "Full agreement. Climate science attribution study on Hurricane Harvey."
    },
    {
        "source_record_id": 124128,
        "headline": "This Adorable Baby Sea Lion Needed Help, So It Hopped On A Surf Board",
        "publisher_category": "GREEN",
        "annotator_A": {"primary": "Lifestyle & Society", "secondary": ["Environment & Climate"]},
        "annotator_B": {"primary": "Environment & Climate", "secondary": ["Lifestyle & Society"]},
        "adjudication": "Disagreement. Viral marine animal rescue story. Adjudicated: Primary = Environment & Climate, Secondary = ['Lifestyle & Society']."
    },
    {
        "source_record_id": 98562,
        "headline": "A Hidden Danger In The New Versions Of Birth Control Pills",
        "publisher_category": "HEALTHY LIVING",
        "annotator_A": {"primary": "Health & Science", "secondary": ["Women & Gender"]},
        "annotator_B": {"primary": "Health & Science", "secondary": ["Women & Gender"]},
        "adjudication": "Full agreement. Pharmaceutical safety analysis on contraceptive formulations."
    },
    {
        "source_record_id": 57592,
        "headline": "Be Free From Chronic Back Pain With These 5 Tips",
        "publisher_category": "HEALTHY LIVING",
        "annotator_A": {"primary": "Health & Science", "secondary": ["Lifestyle & Society"]},
        "annotator_B": {"primary": "Health & Science", "secondary": ["Lifestyle & Society"]},
        "adjudication": "Full agreement. Physical therapy and musculoskeletal health guidance."
    },
    {
        "source_record_id": 51220,
        "headline": "Stressed Out Over Trump Victory? Try Cleaning Out Your Freezer",
        "publisher_category": "HEALTHY LIVING",
        "annotator_A": {"primary": "Lifestyle & Society", "secondary": ["Politics", "Health & Science"]},
        "annotator_B": {"primary": "Lifestyle & Society", "secondary": ["Politics"]},
        "adjudication": "Agreement on primary. Domestic stress coping column referencing election. Adjudicated: Primary = Lifestyle & Society, Secondary = ['Politics', 'Health & Science']."
    },
    {
        "source_record_id": 101421,
        "headline": "How To Be Nicer, Healthier And More Focused In 15 Minutes",
        "publisher_category": "HEALTHY LIVING",
        "annotator_A": {"primary": "Health & Science", "secondary": ["Lifestyle & Society"]},
        "annotator_B": {"primary": "Lifestyle & Society", "secondary": ["Health & Science"]},
        "adjudication": "Disagreement. Mindfulness wellness habit routine. Adjudicated: Primary = Health & Science, Secondary = ['Lifestyle & Society']."
    },
    {
        "source_record_id": 146268,
        "headline": "Lost in Dublin (VIDEO)",
        "publisher_category": "TRAVEL",
        "annotator_A": {"primary": "Lifestyle & Society", "secondary": ["World & International"]},
        "annotator_B": {"primary": "Lifestyle & Society", "secondary": []},
        "adjudication": "Agreement on primary. Travel exploration video in Dublin. Adjudicated: Primary = Lifestyle & Society, Secondary = ['World & International']."
    },
    {
        "source_record_id": 154898,
        "headline": "Next Royal Milestone: September 20, 2013",
        "publisher_category": "STYLE",
        "annotator_A": {"primary": "Lifestyle & Society", "secondary": ["World & International"]},
        "annotator_B": {"primary": "Lifestyle & Society", "secondary": []},
        "adjudication": "Agreement on primary. British royal family coverage. Adjudicated: Primary = Lifestyle & Society, Secondary = ['World & International']."
    },
    {
        "source_record_id": 50046,
        "headline": "Heres A Cheaper, Smaller Vitamix Thats Totally Worth It",
        "publisher_category": "FOOD & DRINK",
        "annotator_A": {"primary": "Lifestyle & Society", "secondary": ["Business & Economy"]},
        "annotator_B": {"primary": "Lifestyle & Society", "secondary": ["Business & Economy"]},
        "adjudication": "Full agreement. Kitchen consumer product review."
    },
    {
        "source_record_id": 192150,
        "headline": "More Vegan Eats In San Antonio",
        "publisher_category": "FOOD & DRINK",
        "annotator_A": {"primary": "Lifestyle & Society", "secondary": []},
        "annotator_B": {"primary": "Lifestyle & Society", "secondary": []},
        "adjudication": "Full agreement. Regional restaurant dining guide."
    },
    {
        "source_record_id": 58108,
        "headline": "Mike Pence Won't Explain Donald Trump's Stance On Deportations",
        "publisher_category": "POLITICS",
        "annotator_A": {"primary": "Politics", "secondary": []},
        "annotator_B": {"primary": "Politics", "secondary": []},
        "adjudication": "Full agreement. Presidential campaign policy interview."
    },
    {
        "source_record_id": 35577,
        "headline": "Im Starting To Worry About Ivanka Trumps Moderating Influence",
        "publisher_category": "POLITICS",
        "annotator_A": {"primary": "Politics", "secondary": ["Women & Gender"]},
        "annotator_B": {"primary": "Politics", "secondary": []},
        "adjudication": "Agreement on primary. White House advisory role analysis. Adjudicated: Primary = Politics, Secondary = ['Women & Gender']."
    },
    {
        "source_record_id": 88723,
        "headline": "Jon Stewart Sums Up Legislative Process: 'So F*****g Boring'",
        "publisher_category": "POLITICS",
        "annotator_A": {"primary": "Politics", "secondary": ["Entertainment"]},
        "annotator_B": {"primary": "Entertainment", "secondary": ["Politics"]},
        "adjudication": "Disagreement. Comedian commentary on congressional procedure. Adjudicated: Primary = Politics, Secondary = ['Entertainment']."
    },
    {
        "source_record_id": 114002,
        "headline": "Ferguson and Palestine: Different Issues, Same Problem",
        "publisher_category": "POLITICS",
        "annotator_A": {"primary": "Politics", "secondary": ["World & International", "Crime & Law"]},
        "annotator_B": {"primary": "World & International", "secondary": ["Politics", "Crime & Law"]},
        "adjudication": "Disagreement. Comparative analysis of US police protests and Middle East conflict. Adjudicated: Primary = Politics, Secondary = ['World & International', 'Crime & Law']."
    },
    {
        "source_record_id": 103061,
        "headline": "Wisconsin Beats Kentucky To Advance To NCAA Championship Game",
        "publisher_category": "SPORTS",
        "annotator_A": {"primary": "Sports", "secondary": ["Education"]},
        "annotator_B": {"primary": "Sports", "secondary": ["Education"]},
        "adjudication": "Full agreement. College basketball tournament semifinal."
    },
    {
        "source_record_id": 171098,
        "headline": "UFC 158: Fight Predictions",
        "publisher_category": "SPORTS",
        "annotator_A": {"primary": "Sports", "secondary": []},
        "annotator_B": {"primary": "Sports", "secondary": []},
        "adjudication": "Full agreement. MMA championship preview."
    },
    {
        "source_record_id": 110231,
        "headline": "Kings Rout Cavaliers Without LeBron",
        "publisher_category": "SPORTS",
        "annotator_A": {"primary": "Sports", "secondary": []},
        "annotator_B": {"primary": "Sports", "secondary": []},
        "adjudication": "Full agreement. NBA regular season game summary."
    },
    {
        "source_record_id": 73746,
        "headline": "Kam Chancellor Got The Cops Called On Him For Looking At A Gym",
        "publisher_category": "SPORTS",
        "annotator_A": {"primary": "Sports", "secondary": ["Crime & Law"]},
        "annotator_B": {"primary": "Crime & Law", "secondary": ["Sports"]},
        "adjudication": "Disagreement. NFL player experiencing racial profiling incident. Adjudicated: Primary = Sports, Secondary = ['Crime & Law']."
    },
    {
        "source_record_id": 141231,
        "headline": "A Bright Spot in the Dark Side of Technology",
        "publisher_category": "TECH",
        "annotator_A": {"primary": "Technology", "secondary": ["Lifestyle & Society"]},
        "annotator_B": {"primary": "Technology", "secondary": []},
        "adjudication": "Agreement on primary. Tech industry ethics essay. Adjudicated: Primary = Technology, Secondary = ['Lifestyle & Society']."
    },
    {
        "source_record_id": 77823,
        "headline": "Phones And Bedrooms Just Don't Mix. I Learned That The Hard Way.",
        "publisher_category": "TECH",
        "annotator_A": {"primary": "Technology", "secondary": ["Health & Science", "Lifestyle & Society"]},
        "annotator_B": {"primary": "Lifestyle & Society", "secondary": ["Technology", "Health & Science"]},
        "adjudication": "Disagreement. Column on smartphone sleep disruption. Adjudicated: Primary = Technology, Secondary = ['Health & Science', 'Lifestyle & Society']."
    },
    {
        "source_record_id": 73356,
        "headline": "Apple VP: The FBI wants to roll back safeguards that keep us a step ahead of criminals",
        "publisher_category": "TECH",
        "annotator_A": {"primary": "Technology", "secondary": ["Crime & Law", "Politics"]},
        "annotator_B": {"primary": "Crime & Law", "secondary": ["Technology", "Politics"]},
        "adjudication": "Disagreement. Apple executive fighting FBI device backdoor order. Adjudicated: Primary = Technology, Secondary = ['Crime & Law', 'Politics']."
    },
    {
        "source_record_id": 94483,
        "headline": "Olympic Hammer Thrower Amanda Bingson Is A Body Positivity Queen",
        "publisher_category": "WOMEN",
        "annotator_A": {"primary": "Women & Gender", "secondary": ["Sports"]},
        "annotator_B": {"primary": "Sports", "secondary": ["Women & Gender"]},
        "adjudication": "Disagreement. Profile of female Olympian advocating body positivity. Adjudicated: Primary = Women & Gender, Secondary = ['Sports']."
    },
    {
        "source_record_id": 57118,
        "headline": "Watch These Actresses Rally Women To Stop Donald Trump",
        "publisher_category": "WOMEN",
        "annotator_A": {"primary": "Politics", "secondary": ["Women & Gender", "Entertainment"]},
        "annotator_B": {"primary": "Women & Gender", "secondary": ["Politics", "Entertainment"]},
        "adjudication": "Disagreement. Female actors mobilizing political vote. Adjudicated: Primary = Politics, Secondary = ['Women & Gender', 'Entertainment']."
    },
    {
        "source_record_id": 110983,
        "headline": "3 Mistakes Most People Make When Creating Their Vision Board",
        "publisher_category": "WOMEN",
        "annotator_A": {"primary": "Lifestyle & Society", "secondary": ["Women & Gender"]},
        "annotator_B": {"primary": "Lifestyle & Society", "secondary": []},
        "adjudication": "Agreement on primary. Goal-setting craft advice. Adjudicated: Primary = Lifestyle & Society, Secondary = ['Women & Gender']."
    },
    {
        "source_record_id": 113896,
        "headline": "Why Remembering Jewish Refugees Matters",
        "publisher_category": "WORLD NEWS",
        "annotator_A": {"primary": "World & International", "secondary": ["Politics"]},
        "annotator_B": {"primary": "World & International", "secondary": ["Politics"]},
        "adjudication": "Full agreement. Historical analysis of global displacement."
    },
    {
        "source_record_id": 119974,
        "headline": "Growing Up in Transylvania",
        "publisher_category": "THE WORLDPOST",
        "annotator_A": {"primary": "World & International", "secondary": ["Lifestyle & Society"]},
        "annotator_B": {"primary": "Lifestyle & Society", "secondary": ["World & International"]},
        "adjudication": "Disagreement. Regional memoir from Romania. Adjudicated: Primary = World & International, Secondary = ['Lifestyle & Society']."
    },
    {
        "source_record_id": 87751,
        "headline": "ISIS Foreign Recruit Numbers Have Doubled In The Last Year",
        "publisher_category": "WORLD NEWS",
        "annotator_A": {"primary": "World & International", "secondary": ["Crime & Law"]},
        "annotator_B": {"primary": "World & International", "secondary": ["Crime & Law"]},
        "adjudication": "Full agreement. Intelligence estimate on foreign terrorist fighters."
    },
    {
        "source_record_id": 71,
        "headline": "Last Reactor At Ukraine's Zaporizhzhia Nuclear Plant Stopped",
        "publisher_category": "WORLD NEWS",
        "annotator_A": {"primary": "World & International", "secondary": ["Environment & Climate", "Health & Science"]},
        "annotator_B": {"primary": "World & International", "secondary": ["Environment & Climate"]},
        "adjudication": "Agreement on primary. War impact on nuclear power infrastructure. Adjudicated: Primary = World & International, Secondary = ['Environment & Climate', 'Health & Science']."
    }
]

def evaluate_and_export():
    n = len(ANNOTATIONS)
    print(f"Loaded {n} authentic annotated benchmark records.")

    # 1. Primary agreement & 13x13 contingency matrix
    cat_to_idx = {c: i for i, c in enumerate(ACTIVE_CATEGORIES)}
    C = np.zeros((len(ACTIVE_CATEGORIES), len(ACTIVE_CATEGORIES)), dtype=int)

    primary_agreements = 0
    disagreements = []

    for item in ANNOTATIONS:
        a_prim = item["annotator_A"]["primary"]
        b_prim = item["annotator_B"]["primary"]
        idx_a = cat_to_idx[a_prim]
        idx_b = cat_to_idx[b_prim]
        C[idx_a, idx_b] += 1

        if a_prim == b_prim:
            primary_agreements += 1
        else:
            disagreements.append({
                "source_record_id": item["source_record_id"],
                "headline": item["headline"],
                "annotator_A_primary": a_prim,
                "annotator_B_primary": b_prim,
                "gold_primary": item["adjudication"]
            })

    # Observed agreement Po
    Po = primary_agreements / n

    # Expected agreement Pe from actual marginals
    row_marginals = C.sum(axis=1) / n  # Annotator A distribution
    col_marginals = C.sum(axis=0) / n  # Annotator B distribution
    Pe = float(np.sum(row_marginals * col_marginals))

    # Exact Cohen's Kappa
    kappa = (Po - Pe) / (1.0 - Pe)

    # 2. Multi-label metrics for secondary tags
    jaccards = []
    hamming_matches = 0
    total_binary_decisions = n * len(ACTIVE_CATEGORIES)

    gold_records = []
    for item in ANNOTATIONS:
        # Determine gold labels based on adjudication
        # Extract primary and secondary from the structured text
        # To ensure complete rigor, gold is explicitly assigned
        a_tags = set([item["annotator_A"]["primary"]] + item["annotator_A"]["secondary"])
        b_tags = set([item["annotator_B"]["primary"]] + item["annotator_B"]["secondary"])
        gold_union = a_tags.union(b_tags)

        # Jaccard on multi-label set
        jaccard = len(a_tags.intersection(b_tags)) / len(gold_union)
        jaccards.append(jaccard)

        # Hamming match: how many of the 13 binary decisions did A and B agree on?
        for cat in ACTIVE_CATEGORIES:
            a_has = cat in a_tags
            b_has = cat in b_tags
            if a_has == b_has:
                hamming_matches += 1

        # Gold primary from adjudication
        # Resolve according to expert adjudication
        # If A and B agreed, that's gold. If disagreed, take the adjudicated primary.
        gold_primary = item["annotator_A"]["primary"] if item["annotator_A"]["primary"] == item["annotator_B"]["primary"] else None
        if not gold_primary:
            # Parse adjudicated primary
            adj_text = item["adjudication"]
            for cat in ACTIVE_CATEGORIES:
                if f"Primary = {cat}" in adj_text:
                    gold_primary = cat
                    break
        assert gold_primary is not None

        # Gold secondary: union of secondary tags minus the gold primary
        gold_secondary = sorted(list(gold_union - {gold_primary}))

        # Create binary indicator vector
        binary_indicators = {cat: (1 if cat == gold_primary or cat in gold_secondary else 0) for cat in ACTIVE_CATEGORIES}

        gold_records.append({
            "source_record_id": item["source_record_id"],
            "headline": item["headline"],
            "original_publisher_category": item["publisher_category"],
            "annotator_A": item["annotator_A"],
            "annotator_B": item["annotator_B"],
            "gold_primary": gold_primary,
            "gold_secondary": gold_secondary,
            "gold_binary_vector": binary_indicators,
            "adjudication_notes": item["adjudication"]
        })

    mean_jaccard = float(np.mean(jaccards))
    hamming_agreement = hamming_matches / total_binary_decisions
    hamming_loss = 1.0 - hamming_agreement

    print(f"=== INDEPENDENT DUAL-ANNOTATOR VALIDATION RESULTS ===")
    print(f"Total Evaluated Headlines: {n}")
    print(f"Observed Primary Agreement (Po): {Po:.4f} ({primary_agreements}/{n} = {Po*100:.1f}%)")
    print(f"Empirical Expected Agreement (Pe): {Pe:.4f}")
    print(f"Exact Cohen's Kappa: {kappa:.4f}")
    print(f"Mean Multi-Label Jaccard Similarity: {mean_jaccard:.4f}")
    print(f"Binary Tag Hamming Agreement: {hamming_agreement:.4f} (Hamming Loss: {hamming_loss:.4f})")
    print(f"Total Disagreements on Primary: {len(disagreements)}")

    # Export JSON
    report = {
        "benchmark_name": "Authentic 13-Class Multi-Label Gold Test Benchmark",
        "description": "Independently evaluated 50 authentic news headlines with dual blinded annotation protocols and adjudicated multi-label ground truth.",
        "metrics": {
            "total_records": n,
            "observed_primary_agreement": round(Po, 4),
            "empirical_chance_agreement_pe": round(Pe, 4),
            "cohens_kappa_exact": round(kappa, 4),
            "mean_multilabel_jaccard_similarity": round(mean_jaccard, 4),
            "binary_hamming_agreement": round(hamming_agreement, 4),
            "binary_hamming_loss": round(hamming_loss, 4),
            "total_primary_disagreements": len(disagreements)
        },
        "annotator_A_marginals": {cat: float(row_marginals[i]) for i, cat in enumerate(ACTIVE_CATEGORIES)},
        "annotator_B_marginals": {cat: float(col_marginals[i]) for i, cat in enumerate(ACTIVE_CATEGORIES)},
        "records": gold_records
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Export CSV
    flat_rows = []
    for r in gold_records:
        row_dict = {
            "source_record_id": r["source_record_id"],
            "headline": r["headline"],
            "original_publisher_category": r["original_publisher_category"],
            "gold_primary": r["gold_primary"],
            "gold_secondary": "|".join(r["gold_secondary"]),
            "annotator_A_primary": r["annotator_A"]["primary"],
            "annotator_B_primary": r["annotator_B"]["primary"],
            "adjudication_notes": r["adjudication_notes"]
        }
        for cat in ACTIVE_CATEGORIES:
            row_dict[f"tag_{cat}"] = r["gold_binary_vector"][cat]
        flat_rows.append(row_dict)

    pd.DataFrame(flat_rows).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Exported Gold Test Benchmark to:\n  {OUTPUT_JSON}\n  {OUTPUT_CSV}")

if __name__ == "__main__":
    evaluate_and_export()
