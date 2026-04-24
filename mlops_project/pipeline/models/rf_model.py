from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from ..config import RANDOM_STATE
from ..data.preprocess import build_preprocessor


@dataclass
class RandomForestPriceModel:
    pipeline: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestPriceModel":
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        max_depth=18,
                        min_samples_leaf=2,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        if self.pipeline is None:
            raise ValueError("Random forest model has not been trained.")
        return self.pipeline.predict(X)
