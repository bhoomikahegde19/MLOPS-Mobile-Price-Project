from __future__ import annotations

import re


class DummyNERModel:
    def fit(self) -> "DummyNERModel":
        return self

    def predict(self, text: str) -> list[dict[str, str | int]]:
        entities: list[dict[str, str | int]] = []
        brand_match = re.search(r"\b(Samsung|Apple|Xiaomi|OnePlus|Motorola|Realme)\b", text, re.IGNORECASE)
        ram_matches = list(re.finditer(r"\b\d+\s?GB RAM\b", text, re.IGNORECASE))
        storage_matches = list(re.finditer(r"\b\d+\s?GB\b", text, re.IGNORECASE))

        if brand_match is not None:
            entities.append(
                {
                    "text": brand_match.group(0),
                    "label": "BRAND",
                    "start": brand_match.start(),
                    "end": brand_match.end(),
                }
            )

        for match in ram_matches:
            entities.append(
                {
                    "text": match.group(0),
                    "label": "RAM",
                    "start": match.start(),
                    "end": match.end(),
                }
            )

        ram_ranges = {(match.start(), match.end()) for match in ram_matches}
        for match in storage_matches:
            if any(match.start() >= start and match.end() <= end for start, end in ram_ranges):
                continue
            entities.append(
                {
                    "text": match.group(0),
                    "label": "STORAGE",
                    "start": match.start(),
                    "end": match.end(),
                }
            )
        entities.sort(key=lambda item: int(item["start"]))
        return entities
