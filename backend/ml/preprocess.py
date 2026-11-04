"""
Text preprocessing module for news headline topic classification.
Provides robust text normalization preserving meaningful domain indicators.
"""

import re
import html

# Custom stopwords list that preserves critical prepositions, negations, and direction words
# Words like 'up', 'down', 'against', 'over', 'under', 'no', 'not' carry high semantic signal in sports/finance headlines.
STANDARD_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
    "which", "this", "that", "these", "those", "then", "just", "so", "than",
    "such", "both", "through", "during", "before", "after", "above", "below",
    "to", "from", "at", "by", "for", "with", "about", "into", "through", "during",
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "would", "should", "could", "ought",
    "i", "you", "he", "she", "it", "we", "they", "them", "their", "theirs",
    "its", "our", "ours", "your", "yours", "his", "her", "hers"
}

# Negations and direction words to explicitly keep
PRESERVE_WORDS = {
    "no", "not", "against", "over", "under", "up", "down", "first", "record",
    "top", "out", "new", "all", "more", "most", "off", "on"
}

FILTER_STOPWORDS = STANDARD_STOPWORDS - PRESERVE_WORDS


def normalize_headline(text: str) -> str:
    """
    Cleans, unescapes, and normalizes a news headline string.
    
    Steps:
    1. Unescape HTML entities (&amp; -> &, &quot; -> ", etc.)
    2. Strip URLs and email addresses
    3. Lowercase text
    4. Remove noise characters while preserving alphanumeric and intra-word hyphens/apostrophes
    5. Filter non-critical stopwords while preserving key directional/negation words
    6. Normalize whitespace
    """
    if not isinstance(text, str):
        return ""

    # HTML entity unescape
    text = html.unescape(text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove HTML tags if any
    text = re.sub(r"<.*?>", " ", text)

    # Lowercase
    text = text.lower()

    # Replace currency symbols with word tokens to preserve financial signal
    text = re.sub(r"\$", " dollar ", text)
    text = re.sub(r"€", " euro ", text)
    text = re.sub(r"£", " pound ", text)
    text = re.sub(r"%", " percent ", text)

    # Remove unwanted punctuation while keeping alphanumeric and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Tokenize and filter selective stopwords
    tokens = text.split()
    filtered_tokens = [w for w in tokens if len(w) > 1 and w not in FILTER_STOPWORDS]

    return " ".join(filtered_tokens)
