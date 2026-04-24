from __future__ import annotations

import json
from pathlib import Path

import joblib

from pipeline.config import REGISTRY_DIR


class ModelLoader:
    def __init__(self) -> None:
        self.production_file = REGISTRY_DIR / "production.json"

    def _read_manifest(self) -> dict:
        if not self.production_file.exists():
            raise FileNotFoundError("Model registry is empty. Run the training pipeline first.")
        with open(self.production_file, "r", encoding="utf-8") as file:
            return json.load(file)

    def load(self, alias: str):
        manifest = self._read_manifest()
        if alias not in manifest:
            raise KeyError(f"Model alias '{alias}' is not registered.")
        artifact_path = Path(manifest[alias]["artifact_path"])
        return joblib.load(artifact_path)
