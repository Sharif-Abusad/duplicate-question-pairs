"""
feature_engineering.py

Feature engineering utilities for Quora Question Pair Similarity.
"""

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from rapidfuzz import fuzz           # replaces fuzzywuzzy
from rapidfuzz import distance     
from difflib import SequenceMatcher
from typing import List

# ------------------------------------------------------------------
# Basic Text Similarity Features
# ------------------------------------------------------------------
def common_words(q1: str, q2: str) -> int:
    """Return count of common words."""
    w1 = set(q1.lower().split())
    w2 = set(q2.lower().split())
    return len(w1.intersection(w2))


def total_words(q1: str, q2: str) -> int:
    """Return total unique words."""
    w1 = set(q1.lower().split())
    w2 = set(q2.lower().split())
    return len(w1) + len(w2)

# ------------------------------------------------------------------
# Token-Based Features
# ------------------------------------------------------------------
def fetch_token_features(q1: str, q2: str) -> List[float]:
    """
    Compute token-based similarity features between two questions.

    Returns:
        List[float]: Eight token similarity features.
    """
    SAFE_DIV = 0.0001

    # STOP_WORDS = pickle.load(open('stopwords','rb'))
    STOP_WORDS = ENGLISH_STOP_WORDS
    token_features = [0.0] * 8

    # Converting the sentence into tokens
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    # Get the non-stopwords in question1
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    # Get the stopwords in question1
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    # Get the common non-stopwords from Question pair
    common_word_count = len(q1_words.intersection(q2_words))

    # Get the common stopwords from Question pair
    common_stop_count = len(q1_stops.intersection(q2_stops))

    # Get the common tokens from Question pair
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)

    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)

    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    # Last word of both questions is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])

    # First word of both questions is same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features

# ------------------------------------------------------------------
# Length-Based Features
# ------------------------------------------------------------------
def fetch_length_features(q1: str, q2: str) -> List[float]:
    """
    Compute length-based similarity features.

    Returns:
        List[float]: Length difference, average length,
        and longest common substring ratio.
    """
    length_features = [0.0] * 3

    # Converting the sentence into tokens
    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    # absolute length feature
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))

    # Average token length of both Questions
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    match = SequenceMatcher(
        None, q1, q2
    ).find_longest_match(
        0, len(q1),
        0, len(q2)
    )

    length_features[2] = match.size

    return length_features

# ------------------------------------------------------------------
# Fuzzy Matching Features
# ------------------------------------------------------------------
def fetch_fuzzy_features(q1: str, q2: str) -> List[float]:
    """
    Compute fuzzy string matching features.

    Returns:
        List[float]: QRatio, Partial Ratio,
        Token Sort Ratio, and Token Set Ratio.
    """

    fuzzy_features = [0.0]*4

    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features
