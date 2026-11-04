# Multi-Label News Headline Annotation Guidelines & Coding Protocols
**Document Version:** 1.0.0-final  
**Target Domain:** 15-Category (13 Active) News Topic Classification System  
**Taxonomy Scope:** Art & Culture, Business & Economy, Crime & Law, Education, Entertainment, Environment & Climate, Health & Science, Lifestyle & Society, Politics, Sports, Technology, Women & Gender, World & International. (Inactive reserved: Automobile & Transport, Weather & Natural Disasters).

---

## 1. Objective

To provide an unambiguous, reproducible decision framework for assigning:
1. **Primary Category:** Exactly one dominant category representing the central subject and core action of the headline.
2. **Secondary Tags:** Zero to three secondary categories that are explicitly involved as secondary themes, institutional stakeholders, or direct consequences.

---

## 2. Taxonomy Definitions

1. **Art & Culture:** Fine art, theater, poetry, architecture, museum exhibitions, photography, literature, and cultural heritage.
2. **Business & Economy:** Corporate finance, markets, macroeconomic indicators (inflation, interest rates), startups, labor strikes, employment, housing markets, and consumer commerce.
3. **Crime & Law:** Criminal acts, police investigations, arrests, court trials, judicial rulings, prison sentencing, law enforcement conduct, and legal statutes.
4. **Education:** K-12 schooling, universities, tuition, academic faculty, student life, campus administration, and pedagogical research.
5. **Entertainment:** Cinema, television, music, pop culture celebrities, comedy, late-night satire, and Hollywood industry events.
6. **Environment & Climate:** Climate change, greenhouse emissions, renewable energy, ocean ecosystems, deforestation, wildlife conservation, and pollution.
7. **Health & Science:** Biomedical clinical research, pharmaceutical drugs, disease outbreaks, neurology, astronomy, physics discoveries, and medical pathology.
8. **Lifestyle & Society:** Food, dining, travel destinations, home living, parenting, personal wellness habits, fashion, personal relationships, and social lifestyle essays.
9. **Politics:** Elections, legislative votes, parliamentary/congressional hearings, political parties, presidential administrations, diplomacy, and partisan campaigning.
10. **Sports:** Professional and amateur athletics, league tournaments, match results, athletic championships, player trades, and athletic profiles.
11. **Technology:** Software, artificial intelligence, cybersecurity, consumer electronics, internet platforms, digital privacy, and telecommunications.
12. **Women & Gender:** Women's reproductive rights, gender wage parity, maternal health policy, female representation in leadership, and systemic gender discrimination.
13. **World & International:** Sovereign geopolitical conflicts, international wars, foreign elections, United Nations treaties, refugee migrations, and international humanitarian crises.

---

## 3. Annotation Coding Protocols

To ensure rigorous boundary analysis on cross-cutting headlines, two distinct protocols are codified and executed in parallel:

### Protocol A: Syntactic Subject & Direct Action Priority Rule
- **Rule A1 (Subject Priority):** Determine the grammatical subject and governing verb. If the subject is an artist or novel, classify as `Art & Culture`. If an athlete or team, classify as `Sports`. If a political candidate, classify as `Politics`.
- **Rule A2 (Direct Action):** The primary category is defined by *what happens directly in the sentence*, not downstream implications.
- **Rule A3 (Secondary Tags):** Assign a secondary tag only when a distinct topical keyword or secondary entity is explicitly named.

### Protocol B: Systemic Institutional Impact Priority Rule (Blinded Pass)
- **Rule B1 (Blinded Context):** Annotator B evaluates the headline with zero visibility into Annotator A's decisions and zero visibility into the original publisher's section tag.
- **Rule B2 (Systemic Impact):** If a headline depicts an event occurring across institutional domains (e.g., a university sexual assault penalty), prioritize the institutional sphere of the core conflict (e.g., `Crime & Law` due to criminal adjudication) over the geographic venue (`Education`).
- **Rule B3 (Secondary Tags):** Capture all impacted stakeholders as secondary tags (up to 3).

---

## 4. Formal Adjudication Rules for Resolving Divergences

When Protocol A and Protocol B diverge on the primary category, adjudication applies the following precedence:
1. **Hard Criminal Actions Precedence:** If the headline describes a felony, indictment, or court trial, legal adjudication takes priority unless the institutional policy context is the sole subject.
2. **Substance Over Satire:** Humorous commentary or late-night jokes targeting political legislation are adjudicated to `Politics` (substantive subject) with `Entertainment` as secondary, unless the article is purely about the performer's show.
3. **Scientific Mechanism vs. Entertainment Medium:** If a headline reports on empirical research explaining human perception of media (e.g., *Study Explains Why Scary Movies Scare Us*), the primary category is `Health & Science`, with `Entertainment` as secondary.
4. **Geopolitical Scope Precedence:** Headlines concerning foreign wars or international displacement are adjudicated to `World & International` with relevant domestic policy tags as secondary.
