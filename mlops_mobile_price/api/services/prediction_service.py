from __future__ import annotations

import pandas as pd

from api.services.model_loader import ModelLoader


class PredictionService:
    def __init__(self) -> None:
        self.loader = ModelLoader()

    def predict(self, features: dict) -> dict:
        model = self.loader.load("linear_regression")
        frame = pd.DataFrame([features])
        prediction = float(model.predict(frame)[0])
        return {"predicted_price": prediction, "model": "linear_regression"}
