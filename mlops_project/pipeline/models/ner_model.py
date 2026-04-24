from __future__ import annotations

import re
from dataclasses import dataclass

import spacy


RAM_PATTERN = re.compile(r"\b(\d{1,2})\s?GB\s?RAM\b", re.IGNORECASE)
STORAGE_PATTERN = re.compile(r"\b(\d{2,4})\s?GB\b", re.IGNORECASE)
BRAND_PATTERN = re.compile(
    r"\b(apple|samsung|xiaomi|oneplus|motorola|realme|oppo|vivo|nokia|google)\b",
    re.IGNORECASE,
)


@dataclass
class MobilePhoneNERModel:
    nlp: object | None = None

    def fit(self) -> "MobilePhoneNERModel":
        self.nlp = spacy.blank("en")
        return self

    def predict(self, text: str) -> list[dict[str, str | int]]:
        if self.nlp is None:
            raise ValueError("NER model has not been initialized.")
        entities: list[dict[str, str | int]] = []
        for match in BRAND_PATTERN.finditer(text):
            entities.append({"text": match.group(0), "label": "BRAND", "start": match.start(), "end": match.end()})
        for match in RAM_PATTERN.finditer(text):
            entities.append({"text": match.group(0), "label": "RAM", "start": match.start(), "end": match.end()})
        for match in STORAGE_PATTERN.finditer(text):
            entities.append({"text": match.group(0), "label": "STORAGE", "start": match.start(), "end": match.end()})
        entities.sort(key=lambda item: int(item["start"]))
        return entities
