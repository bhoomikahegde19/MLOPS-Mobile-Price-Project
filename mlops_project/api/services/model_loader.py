from __future__ import annotations

import json
from pathlib import Path

import joblib

from pipeline.config import REGISTRY_DIR


class ModelLoader:
    def __init__(self):
        self.production_manifest = REGISTRY_DIR / "production.json"

    def _load_manifest(self) -> dict:
        if not self.production_manifest.exists():
            raise FileNotFoundError("Production manifest not found. Run training first.")
        with open(self.production_manifest, "r", encoding="utf-8") as file:
            return json.load(file)

    def load(self, alias: str):
        manifest = self._load_manifest()
        if alias not in manifest:
            raise KeyError(f"Model alias '{alias}' is not registered for production.")
        artifact_path = Path(manifest[alias]["artifact_path"])
        if artifact_path.suffix == ".pt":
            raise ValueError("PyTorch sequence models are registered but not exposed by the API loader.")
        return joblib.load(artifact_path)

    def load_manifest(self) -> dict:
        return self._load_manifest()
