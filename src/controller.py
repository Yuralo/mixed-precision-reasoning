"""Loading/inference helper for the cheap learned sensitivity controller."""

from __future__ import annotations

import joblib
import pandas as pd


def predict_sensitivity(model_path: str, rows: list[dict]) -> list[float]:
    bundle = joblib.load(model_path)
    frame = pd.DataFrame(rows)
    return bundle["model"].predict_proba(frame[bundle["features"]])[:, 1].tolist()
