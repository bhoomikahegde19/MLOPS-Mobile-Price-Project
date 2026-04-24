from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from ..data.preprocess import build_preprocessor


@dataclass
class LinearRegressionModel:
    pipeline: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LinearRegressionModel":
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X)),
                ("model", LinearRegression()),
            ]
        )
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        if self.pipeline is None:
            raise ValueError("Linear regression model has not been trained.")
        return self.pipeline.predict(X)
