from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from pipeline.config import CLASS_LABELS
from pipeline.data.preprocess import build_numeric_preprocessor


@dataclass
class LogisticRegressionModel:
    pipeline: Pipeline | None = None

    def _map_labels(self, y: pd.Series) -> pd.Series:
        return y.map(CLASS_LABELS)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionModel":
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", build_numeric_preprocessor(X.columns.tolist())),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        self.pipeline.fit(X, self._map_labels(y))
        return self

    def predict(self, X: pd.DataFrame):
        if self.pipeline is None:
            raise ValueError("Logistic regression model is not trained.")
        return self.pipeline.predict(X)

    def transform_target(self, y: pd.Series) -> pd.Series:
        return self._map_labels(y)
