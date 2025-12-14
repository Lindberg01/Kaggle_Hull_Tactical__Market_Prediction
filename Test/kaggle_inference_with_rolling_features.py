"""
Kaggle Inference Server with Online Lagged & Rolling Features

This file is a single, extensible template intended for Kaggle time-series /
streaming evaluation competitions.

Key properties:
- Compatible with kaggle_evaluation.default_inference_server
- Computes lagged and rolling-mean features ONLINE (no full-history access)
- Matches offline Pandas behavior: shift(lag).fillna(0) and rolling(mean).fillna(0)
- Designed to be easy to extend with additional features or a trained model
"""

# ------------------------------------------------------------
# Standard Library Imports
# ------------------------------------------------------------
import os
from collections import deque
from typing import Dict, List

# ------------------------------------------------------------
# Third-Party Imports
# ------------------------------------------------------------
import numpy as np
import polars as pl
import kaggle_evaluation.default_inference_server

# ------------------------------------------------------------
# Feature Configuration (EDIT HERE TO ADD MORE FEATURES)
# ------------------------------------------------------------
LAG_PERIODS: List[int] = [1, 3, 5]
ROLL_WINDOW: int = 5

FEATURES_TO_USE: List[str] = [
    "E19",
    "V7",
    "V10",
    "P5",
    "S8",
]

# ------------------------------------------------------------
# Global Rolling State (Persists Across predict() Calls)
# ------------------------------------------------------------
# Each feature keeps a buffer large enough for max lag / rolling window
MAX_BUFFER = max(max(LAG_PERIODS), ROLL_WINDOW)

feature_buffers: Dict[str, deque] = {
    col: deque(maxlen=MAX_BUFFER)
    for col in FEATURES_TO_USE
}

# ------------------------------------------------------------
# Optional: Model Placeholder
# ------------------------------------------------------------
model = None


def load_model():
    """Load and return your trained model.

    This function is intentionally empty and should be filled in later.
    It is safe to call this inside the first predict() invocation.
    """
    return None


# ------------------------------------------------------------
# Predict Function (REQUIRED BY KAGGLE)
# ------------------------------------------------------------
def predict(test: pl.DataFrame) -> pl.DataFrame:
    """
    Called repeatedly by Kaggle's evaluation gateway.

    Parameters
    ----------
    test : pl.DataFrame
        A batch of features for the current timestep(s).

    Returns
    -------
    pl.DataFrame
        Must contain a single column named "prediction".
    """
    global model, feature_buffers

    # Lazy-load model on first call (no time limit on first predict)
    if model is None:
        model = load_model()

    rows = test.to_dicts()
    engineered_rows = []

    for row in rows:
        feats = {}

        for col in FEATURES_TO_USE:
            raw_val = row.get(col)

            # Match pd.to_numeric(errors="coerce")
            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                val = np.nan

            buf = feature_buffers[col]
            buf.append(val)

            # -------------------------
            # Lag Features
            # -------------------------
            for lag in LAG_PERIODS:
                if len(buf) > lag:
                    feats[f"{col}_lag{lag}"] = list(buf)[-lag - 1]
                else:
                    feats[f"{col}_lag{lag}"] = 0.0

            # -------------------------
            # Rolling Mean Feature
            # -------------------------
            if len(buf) >= ROLL_WINDOW:
                feats[f"{col}_rollmean{ROLL_WINDOW}"] = np.nanmean(
                    list(buf)[-ROLL_WINDOW:]
                )
            else:
                feats[f"{col}_rollmean{ROLL_WINDOW}"] = 0.0

        engineered_rows.append(feats)

    feature_df = pl.DataFrame(engineered_rows)

    # --------------------------------------------------------
    # Model Inference (PLACEHOLDER)
    # --------------------------------------------------------
    # Replace this block with your actual model prediction
    # Example:
    # preds = model.predict(feature_df.to_numpy())

    preds = pl.Series("prediction", [0.0] * len(feature_df))

    return pl.DataFrame({"prediction": preds})


# ------------------------------------------------------------
# Inference Server Bootstrapping (DO NOT MODIFY)
# ------------------------------------------------------------
inference_server = kaggle_evaluation.default_inference_server.DefaultInferenceServer(
    predict
)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        ("/kaggle/input/hull-tactical-market-prediction/",)
    )
