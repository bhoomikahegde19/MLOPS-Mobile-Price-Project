from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pipeline.config import MODEL_DIR, REGISTRY_DIR


class ModelRegistry:
    def __init__(self) -> None:
        self.registry_file = REGISTRY_DIR / "registry.json"
        self.production_file = REGISTRY_DIR / "production.json"
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    def _load_json(self, path: Path, default: dict) -> dict:
        if path.exists():
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        return default

    def _save_json(self, path: Path, payload: dict) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def save_model(self, model_name: str, source_path: str, alias: str | None = None) -> dict:
        registry = self._load_json(self.registry_file, {"models": {}})
        version = f"v{len(registry['models'].get(model_name, [])) + 1}"
        source = Path(source_path)
        target_dir = REGISTRY_DIR / model_name / version
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source.name
        shutil.copy2(source, target_path)

        record = {
            "version": version,
            "artifact_path": str(target_path),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        registry["models"].setdefault(model_name, []).append(record)
        self._save_json(self.registry_file, registry)

        production = self._load_json(self.production_file, {})
        production[alias or model_name] = {"model_name": model_name, **record}
        self._save_json(self.production_file, production)
        return production[alias or model_name]
