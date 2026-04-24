from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..data.preprocess import build_preprocessor


@dataclass
class LogisticRegressionClassifier:
    pipeline: Pipeline | None = None
    labels: list[str] = field(default_factory=lambda: ["Low", "Medium", "High"])
    thresholds: list[float] | None = None

    def _to_categories(self, y: pd.Series) -> pd.Series:
        numeric_y = pd.to_numeric(y, errors="coerce")
        unique_values = sorted(value for value in numeric_y.dropna().unique().tolist())

        if unique_values == [0, 1, 2, 3]:
            mapping = {0: "Low", 1: "Medium", 2: "Medium", 3: "High"}
            return numeric_y.map(mapping)

        if self.thresholds is None:
            q1, q2 = numeric_y.quantile([0.33, 0.66]).tolist()
            self.thresholds = [float(q1), float(q2)]
        bins = [-np.inf, self.thresholds[0], self.thresholds[1], np.inf]
        return pd.cut(numeric_y, bins=bins, labels=self.labels, include_lowest=True)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionClassifier":
        y_cat = self._to_categories(y)
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X)),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )
        self.pipeline.fit(X, y_cat)
        return self

    def predict(self, X: pd.DataFrame):
        if self.pipeline is None:
            raise ValueError("Logistic regression model has not been trained.")
        return self.pipeline.predict(X)

    def transform_target(self, y: pd.Series) -> pd.Series:
        return self._to_categories(y)
