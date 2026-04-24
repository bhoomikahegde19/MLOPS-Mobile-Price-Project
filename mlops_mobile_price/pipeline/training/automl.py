from __future__ import annotations

import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from pipeline.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_DIR, REPORTS_DIR
from pipeline.models.linear_regression import LinearRegressionModel
from pipeline.models.logistic_regression import LogisticRegressionModel
from pipeline.models.lstm_model import LSTMPlaceholderModel
from pipeline.models.ner_model import DummyNERModel
from pipeline.models.rf_model import RandomForestModel
from pipeline.training.evaluate import evaluate_classification, evaluate_regression


class AutoMLTrainer:
    def __init__(self) -> None:
        self.regression_models = {
            "linear_regression": LinearRegressionModel(),
            "random_forest": RandomForestModel(),
            "lstm_model": LSTMPlaceholderModel(),
        }
        self.classifier = LogisticRegressionModel()
        self.ner_model = DummyNERModel()

    def train_all(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        results = {"regression": {}, "classification": {}, "ner": {}}
        best_model_name = None
        best_rmse = float("inf")

        for model_name, model in self.regression_models.items():
            with mlflow.start_run(run_name=model_name):
                trained_model = model.fit(X_train, y_train)
                predictions = trained_model.predict(X_test)
                metrics = evaluate_regression(y_test, predictions)
                artifact_path = MODEL_DIR / f"{model_name}.joblib"
                joblib.dump(trained_model, artifact_path)

                mlflow.log_params({"model_name": model_name, "task": "regression"})
                mlflow.log_metrics(metrics)
                if hasattr(trained_model, "pipeline") and trained_model.pipeline is not None:
                    mlflow.sklearn.log_model(sk_model=trained_model.pipeline, name=model_name)
                else:
                    mlflow.log_artifact(str(artifact_path), artifact_path="models")

                results["regression"][model_name] = {
                    "metrics": metrics,
                    "artifact_path": str(artifact_path),
                }
                if metrics["rmse"] < best_rmse:
                    best_rmse = metrics["rmse"]
                    best_model_name = model_name

        with mlflow.start_run(run_name="logistic_regression"):
            classifier = self.classifier.fit(X_train, y_train)
            true_labels = classifier.transform_target(y_test)
            predicted_labels = classifier.predict(X_test)
            metrics = evaluate_classification(true_labels, predicted_labels)
            artifact_path = MODEL_DIR / "logistic_regression.joblib"
            joblib.dump(classifier, artifact_path)

            mlflow.log_params({"model_name": "logistic_regression", "task": "classification"})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(sk_model=classifier.pipeline, name="logistic_regression")

            results["classification"]["logistic_regression"] = {
                "metrics": metrics,
                "artifact_path": str(artifact_path),
            }

        ner_artifact = MODEL_DIR / "ner_model.joblib"
        joblib.dump(self.ner_model.fit(), ner_artifact)
        results["ner"]["ner_model"] = {
            "metrics": {"entity_count_demo": 3},
            "artifact_path": str(ner_artifact),
        }

        summary = {
            "best_model_name": best_model_name,
            "best_model_metrics": results["regression"][best_model_name]["metrics"],
            "all_results": results,
        }
        with open(REPORTS_DIR / "automl_summary.json", "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        return summary
