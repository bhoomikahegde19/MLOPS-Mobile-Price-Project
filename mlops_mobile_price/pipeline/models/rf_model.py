from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from pipeline.config import RANDOM_STATE
from pipeline.data.preprocess import build_numeric_preprocessor


@dataclass
class RandomForestModel:
    pipeline: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", build_numeric_preprocessor(X.columns.tolist())),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        max_depth=16,
                        min_samples_leaf=2,
                        n_jobs=1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        if self.pipeline is None:
            raise ValueError("Random forest model is not trained.")
        return self.pipeline.predict(X)
