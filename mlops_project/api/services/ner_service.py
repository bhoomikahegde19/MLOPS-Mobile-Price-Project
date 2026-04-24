from __future__ import annotations

from .model_loader import ModelLoader


class NERService:
    def __init__(self):
        self.loader = ModelLoader()

    def extract_entities(self, text: str) -> dict:
        model = self.loader.load("ner")
        return {"entities": model.predict(text), "model": "ner"}
