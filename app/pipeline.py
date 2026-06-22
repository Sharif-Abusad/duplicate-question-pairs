"""
pipeline.py

Creates the final feature vector for the Quora Question Pair
Similarity model by combining handcrafted features and
Bag-of-Words (BoW) representations.
"""

# ------------------------------------------------------------------
# Import Library
# ------------------------------------------------------------------
import joblib

import numpy as np

from app.preprocessing import preprocess

from app.feature_engineering import (
    common_words,
    total_words,
    fetch_token_features,
    fetch_length_features,
    fetch_fuzzy_features
)
from app.constants import SAFE_DIV, WORD2VEC_PATH

# ------------------------------------------------------------------
# Load Trained CountVectorizer
# ------------------------------------------------------------------

try:
    with open(WORD2VEC_PATH, "rb") as file:
        # W2V = pickle.load(file)
        W2V = joblib.load(WORD2VEC_PATH)

except FileNotFoundError:
    raise FileNotFoundError(
        f"CountVectorizer not found at '{WORD2VEC_PATH}'. "
        "Ensure the model file exists before running the pipeline."
    )

# ------------------------------------------------------------------
# Feature Generation Pipeline
# ------------------------------------------------------------------
def document_vector(doc):
    doc = [word for word in doc.split() if word in W2V.wv.index_to_key]
    return np.mean(W2V.wv[doc], axis=0)

def create_feature_vector(q1: str, q2: str) -> np.ndarray:
    """
    Generate the final model input feature vector.

    Steps:
    1. Preprocess both questions.
    2. Extract handcrafted similarity features.
    3. Generate Bag-of-Words representations.
    4. Combine all features into a single NumPy array.

    Parameters
    ----------
    q1 : str
        First question.

    q2 : str
        Second question.

    Returns
    -------
    np.ndarray
        Final feature vector used for prediction.
    """

    # --------------------------------------------------------------
    # Text Preprocessing
    # --------------------------------------------------------------

    q1 = preprocess(q1)
    q2 = preprocess(q2)

    # --------------------------------------------------------------
    # Basic Similarity Features
    # --------------------------------------------------------------

    common_word_count = common_words(q1, q2)
    total_word_count = total_words(q1, q2)

    basic_features = [
        len(q1),
        len(q2),
        len(q1.split()),
        len(q2.split()),
        common_word_count,
        total_word_count,
        round(
            common_word_count /
            (total_word_count + SAFE_DIV),
            2
        )
    ]

    # --------------------------------------------------------------
    # Token-Based Features
    # --------------------------------------------------------------

    token_features = fetch_token_features(q1, q2)

    # --------------------------------------------------------------
    # Length-Based Features
    # --------------------------------------------------------------

    length_features = fetch_length_features(q1, q2)

    # --------------------------------------------------------------
    # Fuzzy Matching Features
    # --------------------------------------------------------------

    fuzzy_features = fetch_fuzzy_features(q1, q2)

    # --------------------------------------------------------------
    # Combine Handcrafted Features
    # --------------------------------------------------------------

    handcrafted_features = (
        basic_features
        + token_features
        + length_features
        + fuzzy_features
    )

    # Total handcrafted features = 22

    handcrafted_features = np.array(
        handcrafted_features
    ).reshape(1, 22)

    # --------------------------------------------------------------
    # Bag-of-Words Features
    # --------------------------------------------------------------
    
    q1_w2v = document_vector(q1)
    q2_w2v = document_vector(q2)

    q1_w2v = q1_w2v.reshape(1, -1)
    q2_w2v = q2_w2v.reshape(1, -1)

    # --------------------------------------------------------------
    # Final Feature Vector
    # --------------------------------------------------------------
    
    return np.hstack(
        (
            q1_w2v,
            q2_w2v,
            handcrafted_features
        )
    )