from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LSTMPlaceholderModel:
    mean_target: float | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMPlaceholderModel":
        del X
        self.mean_target = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame):
        if self.mean_target is None:
            raise ValueError("LSTM placeholder model is not trained.")
        return np.full(shape=len(X), fill_value=self.mean_target, dtype=float)
