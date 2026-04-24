from __future__ import annotations

import json

from pipeline.data.loader import load_csv, resolve_dataset_path
from pipeline.registry.model_registry import ModelRegistry
from pipeline.training.automl import AutoMLTrainer
from pipeline.training.dataset import create_train_test_split


def run_training() -> dict:
    dataset_path = resolve_dataset_path()
    df = load_csv()
    split = create_train_test_split(df)

    trainer = AutoMLTrainer()
    results = trainer.train_all(split.X_train, split.X_test, split.y_train, split.y_test)

    registry = ModelRegistry()
    registry.save_model("linear_regression", results["all_results"]["regression"]["linear_regression"]["artifact_path"], alias="linear_regression")
    registry.save_model("random_forest", results["all_results"]["regression"]["random_forest"]["artifact_path"], alias="random_forest")
    registry.save_model("lstm_model", results["all_results"]["regression"]["lstm_model"]["artifact_path"], alias="lstm_model")
    registry.save_model(
        "logistic_regression",
        results["all_results"]["classification"]["logistic_regression"]["artifact_path"],
        alias="logistic_regression",
    )
    registry.save_model("ner_model", results["all_results"]["ner"]["ner_model"]["artifact_path"], alias="ner_model")
    registry.save_model(
        results["best_model_name"],
        results["all_results"]["regression"][results["best_model_name"]]["artifact_path"],
        alias="production_model",
    )

    results["dataset_path"] = str(dataset_path)
    return results


if __name__ == "__main__":
    print(json.dumps(run_training(), indent=2))
