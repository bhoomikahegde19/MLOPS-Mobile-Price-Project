from __future__ import annotations

import pandas as pd

from api.services.model_loader import ModelLoader


class ClassificationService:
    def __init__(self) -> None:
        self.loader = ModelLoader()

    def classify(self, features: dict) -> dict:
        model = self.loader.load("logistic_regression")
        frame = pd.DataFrame([features])
        prediction = str(model.predict(frame)[0])
        return {"predicted_class": prediction, "model": "logistic_regression"}
