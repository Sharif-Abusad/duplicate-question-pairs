# ==========================================================
# Preprocessing Module
#
# Handles text cleaning and normalization for duplicate
# question detection. Converts raw user input into a
# standardized format suitable for feature extraction,
# similarity computation, and machine learning inference.
# ==========================================================

# ==========================================================
# Library Imports
# ==========================================================
from bs4 import BeautifulSoup
import re
import string
from app.constants import CONTRACTIONS

# ==========================================================
# Text Preprocessing
# Cleans and normalizes question text before feature extraction.
# ==========================================================
def preprocess(q):
    """
    Perform text normalization and cleaning.

    Steps:
    1. Convert text to lowercase.
    2. Replace special symbols with words.
    3. Normalize large numeric values.
    4. Expand contractions.
    5. Remove HTML tags.
    6. Remove punctuation.

    Parameters
    ----------
    q : str
        Input question text.

    Returns
    -------
    str
        Cleaned and normalized text.
    """
    # Convert input to lowercase and remove leading/trailing spaces
    q = str(q).lower().strip()

    # Replace special symbols with their textual representations
    # to preserve semantic meaning during vectorization
    q = q.replace('%', 'percent')
    q = q.replace('$', 'dollar')
    q = q.replace('₹', 'rupee')
    q = q.replace('€', 'euro')
    q = q.replace('@', 'at')

    # Remove dataset-specific [math] tokens
    q = q.replace('[math]', '')

    # Normalize large numeric values
    # Example: 1,000 -> 1k, 1,000,000 -> 1m
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')

    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # Expand English contractions to their full forms
    # Example: can't -> cannot, I'm -> I am

    # Replace contractions in the text
    q_decontracted = []
    for word in q.split():
        q_decontracted.append(
            CONTRACTIONS.get(word, word)
        )

    q = ' '.join(q_decontracted)

    # Handle remaining contraction patterns
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    # Remove HTML tags and extract clean text
    q = BeautifulSoup(q, "html.parser")
    q = q.get_text()

    # Remove punctuation symbols
    # Example: "hello?" -> "hello"
    exclude = string.punctuation
    q = q.translate(str.maketrans('', '', exclude))

    return q