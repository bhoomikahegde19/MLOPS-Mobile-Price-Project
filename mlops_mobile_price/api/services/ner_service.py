from __future__ import annotations

from api.services.model_loader import ModelLoader


class NERService:
    def __init__(self) -> None:
        self.loader = ModelLoader()

    def extract(self, text: str) -> dict:
        model = self.loader.load("ner_model")
        return {"entities": model.predict(text), "model": "ner_model"}
