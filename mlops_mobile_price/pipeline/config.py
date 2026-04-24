from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
MLRUNS_DIR = EXPERIMENTS_DIR / "mlruns"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
REGISTRY_DIR = ARTIFACTS_DIR / "registry"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

DATASET_PATH = Path(os.getenv("DATASET_PATH", DATA_DIR / "train.csv"))
KAGGLE_DATASET_SLUG = os.getenv("KAGGLE_DATASET_SLUG", "dewangmoghe/mobile-phone-price-prediction")
PUBLIC_DATASET_FALLBACK_URL = os.getenv(
    "PUBLIC_DATASET_FALLBACK_URL",
    "https://raw.githubusercontent.com/Datamanim/datarepo/main/mobile/train.csv",
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", MLRUNS_DIR.resolve().as_uri())
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlops-mobile-price")
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))

TARGET_COLUMN = "price_range"
CLASS_LABELS = {0: "Low", 1: "Medium", 2: "Medium", 3: "High"}
FEATURE_COLUMNS = [
    "battery_power",
    "blue",
    "clock_speed",
    "dual_sim",
    "fc",
    "four_g",
    "int_memory",
    "m_dep",
    "mobile_wt",
    "n_cores",
    "pc",
    "px_height",
    "px_width",
    "ram",
    "sc_h",
    "sc_w",
    "talk_time",
    "three_g",
    "touch_screen",
    "wifi",
]
