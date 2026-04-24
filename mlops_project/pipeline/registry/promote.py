from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..config import MODEL_STORE_DIR, REGISTRY_DIR


class ModelRegistry:
    def __init__(self):
        self.registry_manifest = REGISTRY_DIR / "registry.json"
        self.production_manifest = REGISTRY_DIR / "production.json"
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)

    def _load_json(self, path: Path, default):
        if path.exists():
            with open(path, "r", encoding="utf-8") as file:
                return json.load(file)
        return default

    def _save_json(self, path: Path, payload) -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def _next_version(self, model_name: str) -> str:
        manifest = self._load_json(self.registry_manifest, {"models": {}})
        return f"v{len(manifest['models'].get(model_name, [])) + 1}"

    def promote(self, model_name: str, source_path: str, task: str = "regression", production_alias: str | None = None) -> dict:
        version = self._next_version(model_name)
        source = Path(source_path)
        target_dir = REGISTRY_DIR / model_name / version
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / source.name
        shutil.copy2(source, target_path)

        manifest = self._load_json(self.registry_manifest, {"models": {}})
        manifest["models"].setdefault(model_name, []).append(
            {
                "version": version,
                "task": task,
                "artifact_path": str(target_path),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        self._save_json(self.registry_manifest, manifest)

        alias = production_alias or model_name
        production = self._load_json(self.production_manifest, {})
        production[alias] = {
            "model_name": model_name,
            "version": version,
            "task": task,
            "artifact_path": str(target_path),
        }
        self._save_json(self.production_manifest, production)
        return production[alias]
