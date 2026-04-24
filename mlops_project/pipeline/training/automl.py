from __future__ import annotations

import json
import shutil

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from ..config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_NAMES, MODEL_STORE_DIR, REPORTS_DIR
from ..models.linear_regression import LinearRegressionModel
from ..models.logistic_regression import LogisticRegressionClassifier
from ..models.lstm_model import LSTMPriceModel
from ..models.ner_model import MobilePhoneNERModel
from ..models.rf_model import RandomForestPriceModel
from .evaluate import classification_metrics, regression_metrics


class AutoMLTrainer:
    def __init__(self):
        self.regression_candidates = {
            MODEL_NAMES["linear_regression"]: LinearRegressionModel(),
            MODEL_NAMES["random_forest"]: RandomForestPriceModel(),
            MODEL_NAMES["lstm"]: LSTMPriceModel(),
        }
        self.classification_model = LogisticRegressionClassifier()
        self.ner_model = MobilePhoneNERModel()

    def _ensure_dirs(self) -> None:
        MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def train_all(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict:
        self._ensure_dirs()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        results: dict[str, dict] = {"regression": {}, "classification": {}, "ner": {}}
        best_regression_model = None
        best_regression_name = None
        best_rmse = float("inf")

        for model_name, model in self.regression_candidates.items():
            with mlflow.start_run(run_name=model_name):
                trained_model = model.fit(X_train, y_train)
                predictions = trained_model.predict(X_test)
                metrics = regression_metrics(y_test, predictions)
                mlflow.log_params({"model_name": model_name, "task": "regression"})
                mlflow.log_metrics(metrics)

                if model_name == MODEL_NAMES["lstm"]:
                    import torch

                    model_path = MODEL_STORE_DIR / f"{model_name}.pt"
                    torch.save(
                        {
                            "state_dict": trained_model.model.state_dict(),
                            "preprocessor": trained_model.preprocessor,
                            "epochs": trained_model.epochs,
                        },
                        model_path,
                    )
                    mlflow.log_artifact(str(model_path), artifact_path="models")
                else:
                    model_path = MODEL_STORE_DIR / f"{model_name}.pkl"
                    joblib.dump(trained_model, model_path)
                    mlflow.sklearn.log_model(sk_model=trained_model.pipeline, name=model_name)

                results["regression"][model_name] = {"metrics": metrics, "artifact_path": str(model_path)}
                if metrics["rmse"] < best_rmse:
                    best_rmse = metrics["rmse"]
                    best_regression_name = model_name
                    best_regression_model = trained_model

        with mlflow.start_run(run_name=MODEL_NAMES["logistic_regression"]):
            clf = self.classification_model.fit(X_train, y_train)
            y_test_labels = clf.transform_target(y_test)
            predictions = clf.predict(X_test)
            metrics = classification_metrics(y_test_labels, predictions)
            mlflow.log_params({"model_name": MODEL_NAMES["logistic_regression"], "task": "classification"})
            mlflow.log_metrics({"accuracy": float(metrics["accuracy"]), "macro_f1": float(metrics["macro_f1"])})
            logistic_path = MODEL_STORE_DIR / "logistic_regression.pkl"
            joblib.dump(clf, logistic_path)
            mlflow.sklearn.log_model(sk_model=clf.pipeline, name="logistic_regression")
            results["classification"][MODEL_NAMES["logistic_regression"]] = {
                "metrics": metrics,
                "artifact_path": str(logistic_path),
            }

        with mlflow.start_run(run_name=MODEL_NAMES["ner"]):
            ner_model = self.ner_model.fit()
            sample_text = "Samsung phone with 8GB RAM and 256GB storage"
            sample_entities = ner_model.predict(sample_text)
            mlflow.log_params({"model_name": MODEL_NAMES["ner"], "task": "ner"})
            mlflow.log_metric("entity_count_demo", float(len(sample_entities)))
            ner_dir = MODEL_STORE_DIR / MODEL_NAMES["ner"]
            if ner_dir.exists():
                shutil.rmtree(ner_dir)
            ner_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(ner_model, ner_dir / "ner_model.pkl")
            with open(ner_dir / "metadata.json", "w", encoding="utf-8") as file:
                json.dump({"sample_entities": sample_entities}, file, indent=2)
            mlflow.log_artifacts(str(ner_dir), artifact_path="models/ner")
            results["ner"][MODEL_NAMES["ner"]] = {
                "metrics": {"entity_count_demo": len(sample_entities)},
                "artifact_path": str(ner_dir / "ner_model.pkl"),
            }

        summary = {
            "best_model_name": best_regression_name,
            "best_model_metrics": results["regression"][best_regression_name]["metrics"] if best_regression_name else {},
            "best_model_object": best_regression_model.__class__.__name__ if best_regression_model else None,
            "feature_columns": X_train.columns.tolist(),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "all_results": results,
        }
        with open(REPORTS_DIR / "automl_summary.json", "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        return summary
