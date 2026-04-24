from __future__ import annotations

import json

from ..data.loader import infer_target_column, load_dataset, resolve_dataset_path
from ..data.preprocess import split_data
from ..registry.promote import ModelRegistry
from .automl import AutoMLTrainer


def run_training() -> dict:
    dataset_path = resolve_dataset_path()
    df = load_dataset(dataset_path)
    target_column = infer_target_column(df)
    split = split_data(df, target_column)

    automl = AutoMLTrainer()
    results = automl.train_all(split.X_train, split.X_test, split.y_train, split.y_test)

    registry = ModelRegistry()
    registry.promote("linear_regression", results["all_results"]["regression"]["linear_regression"]["artifact_path"], task="regression")
    registry.promote("logistic_regression", results["all_results"]["classification"]["logistic_regression"]["artifact_path"], task="classification")
    registry.promote("random_forest", results["all_results"]["regression"]["random_forest"]["artifact_path"], task="regression")
    registry.promote("lstm", results["all_results"]["regression"]["lstm"]["artifact_path"], task="regression", production_alias="best_sequence_model")
    registry.promote("ner", results["all_results"]["ner"]["ner"]["artifact_path"], task="ner")

    best_name = results["best_model_name"]
    registry.promote(best_name, results["all_results"]["regression"][best_name]["artifact_path"], task="regression", production_alias="production_model")
    results["dataset_path"] = str(dataset_path)
    results["target_column"] = target_column
    return results


if __name__ == "__main__":
    print(json.dumps(run_training(), indent=2))
