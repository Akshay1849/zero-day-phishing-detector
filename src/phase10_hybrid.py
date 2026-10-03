from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class Phase10HybridModel:
    """Phase 10 hybrid wrapper preserving raw labels: 0 = phishing, 1 = legitimate."""

    classes_ = np.array([0, 1], dtype=int)

    def __init__(
        self,
        rf_model: Any,
        svm_pipeline: Any,
        feature_order: list[str],
        rf_path: str | Path | None = None,
        svm_path: str | Path | None = None,
    ) -> None:
        self.rf_model = rf_model
        self.svm_pipeline = svm_pipeline
        self.feature_order = list(feature_order)
        self.rf_path = str(rf_path) if rf_path is not None else None
        self.svm_path = str(svm_path) if svm_path is not None else None
        self.weights = (0.5, 0.5)

    @staticmethod
    def _phishing_probability(model: Any, frame: pd.DataFrame) -> float:
        probabilities = model.predict_proba(frame)
        classes = list(model.classes_)
        if 0 not in classes:
            raise ValueError("Model does not contain raw phishing class 0.")
        return float(probabilities[0][classes.index(0)])

    @staticmethod
    def _normalize_probability_value(value: float) -> float:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("Probability must be finite.")
        return min(max(value, 0.0), 1.0)

    def _frame(self, values: list[float | int] | pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(values, pd.DataFrame):
            frame = values.copy()
        else:
            frame = pd.DataFrame([list(values)], columns=self.feature_order)
        return frame

    def predict_with_details(self, values: list[float | int] | pd.DataFrame | np.ndarray) -> tuple[int, float, float, float, float]:
        frame = self._frame(values)
        rf_probability = self._phishing_probability(self.rf_model, frame)
        svm_probability = self._phishing_probability(self.svm_pipeline, frame)
        hybrid_probability = self.weights[0] * rf_probability + self.weights[1] * svm_probability
        hybrid_probability = self._normalize_probability_value(hybrid_probability)
        prediction = 0 if hybrid_probability >= 0.5 else 1
        confidence = float(hybrid_probability if prediction == 0 else 1.0 - hybrid_probability)
        return int(prediction), float(confidence), float(rf_probability), float(svm_probability), float(hybrid_probability)

    def predict(self, values: list[float | int] | pd.DataFrame | np.ndarray) -> np.ndarray:
        prediction, _, _, _, _ = self.predict_with_details(values)
        return np.array([int(prediction)], dtype=int)

    def predict_proba(self, values: list[float | int] | pd.DataFrame | np.ndarray) -> np.ndarray:
        frame = self._frame(values)
        rf_probability = self._phishing_probability(self.rf_model, frame)
        svm_probability = self._phishing_probability(self.svm_pipeline, frame)
        hybrid_probability = self.weights[0] * rf_probability + self.weights[1] * svm_probability
        hybrid_probability = self._normalize_probability_value(hybrid_probability)
        legitimate_probability = 1.0 - hybrid_probability
        return np.array([[hybrid_probability, legitimate_probability]], dtype=float)

    def __call__(self, values: list[float | int] | pd.DataFrame | np.ndarray):
        return self.predict(values)
