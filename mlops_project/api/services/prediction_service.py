from __future__ import annotations

import pandas as pd

from .model_loader import ModelLoader


class PredictionService:
    def __init__(self):
        self.loader = ModelLoader()

    def predict_price(self, features: dict) -> dict:
        model = self.loader.load("linear_regression")
        frame = pd.DataFrame([features])
        return {"predicted_price": float(model.predict(frame)[0]), "model": "linear_regression"}

    def classify_price_band(self, features: dict) -> dict:
        model = self.loader.load("logistic_regression")
        frame = pd.DataFrame([features])
        return {"predicted_class": str(model.predict(frame)[0]), "model": "logistic_regression"}
