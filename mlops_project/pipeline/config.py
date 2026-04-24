from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
MLRUNS_DIR = EXPERIMENTS_DIR / "mlruns"
REGISTRY_DIR = PROJECT_ROOT / "artifacts" / "registry"
MODEL_STORE_DIR = PROJECT_ROOT / "artifacts" / "models"
REPORTS_DIR = PROJECT_ROOT / "artifacts" / "reports"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DATASET_PATH = DEFAULT_DATA_DIR / "train.csv"
KAGGLE_DATASET_SLUG = os.getenv("KAGGLE_DATASET_SLUG", "dewangmoghe/mobile-phone-price-prediction")
PUBLIC_DATASET_FALLBACK_URL = os.getenv(
    "PUBLIC_DATASET_FALLBACK_URL",
    "https://raw.githubusercontent.com/Datamanim/datarepo/main/mobile/train.csv",
)

DATASET_PATH = Path(os.getenv("DATASET_PATH", DEFAULT_DATASET_PATH))
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mobile-phone-price-prediction")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", MLRUNS_DIR.resolve().as_uri())
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
VALIDATION_SIZE = float(os.getenv("VALIDATION_SIZE", "0.1"))
MODEL_SELECTION_PRIMARY_METRIC = os.getenv("MODEL_SELECTION_PRIMARY_METRIC", "rmse")

MODEL_NAMES = {
    "linear_regression": "linear_regression",
    "logistic_regression": "logistic_regression",
    "random_forest": "random_forest",
    "lstm": "lstm",
    "ner": "ner",
}

PRICE_TARGET_CANDIDATES = [
    "price_range",
    "price",
    "Price",
    "Price Range",
    "price_usd",
    "selling_price",
]

BRAND_CANDIDATES = ["brand", "Brand", "company", "Company", "manufacturer"]
MOBILE_PRICE_CLASS_COLUMN = "price_range"
