import json
import os
import joblib

from app.constants import MODEL_PATH

# ------------------------------------------------------------------
# Load trained model
# ------------------------------------------------------------------
# The model is loaded once at application startup to avoid repeated
# disk I/O and improve prediction latency.
# ------------------------------------------------------------------
model = joblib.load(MODEL_PATH)

# ------------------------------------------------------------------
# Load model metadata
# ------------------------------------------------------------------
# Metadata contains deployment-specific settings such as the
# optimized classification threshold selected during model evaluation.
# ------------------------------------------------------------------
XGB_META = os.path.join(
    os.path.dirname(MODEL_PATH),
    "model_meta.json"
)

with open(XGB_META, "r") as file:
    meta = json.load(file)

BEST_THRESHOLD = meta["best_threshold"]


def predict_duplicate(x, threshold: float = BEST_THRESHOLD) -> dict:
    """
    Predict whether two questions are duplicates.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        Feature vector generated from a question pair.

    threshold : float, default=BEST_THRESHOLD
        Probability threshold used to convert predicted
        probabilities into a binary duplicate/non-duplicate label.

    Returns
    -------
    dict
        Dictionary containing:

        - duplicate : bool
            Predicted duplicate status.

        - probability : float
            Predicted probability of being a duplicate question pair.

        - confidence : str
            Confidence level associated with the prediction
            ('high', 'medium', or 'low').
    """

    # Generate duplicate probability score
    prob = model.predict_proba(x)[0][1]

    # Convert probability into binary prediction
    is_duplicate = prob >= threshold

    # Assign confidence level based on probability
    if prob > 0.75:
        confidence = "high"
    elif prob > 0.55:
        confidence = "medium"
    elif prob < 0.25:
        confidence = "high"
    else:
        confidence = "low"

    return {
        "duplicate": bool(is_duplicate),
        "probability": round(float(prob), 4),
        "confidence": confidence
    }