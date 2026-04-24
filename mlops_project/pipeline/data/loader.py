from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..config import (
    BRAND_CANDIDATES,
    DATASET_PATH,
    DEFAULT_DATA_DIR,
    KAGGLE_DATASET_SLUG,
    MOBILE_PRICE_CLASS_COLUMN,
    PRICE_TARGET_CANDIDATES,
    PUBLIC_DATASET_FALLBACK_URL,
)


def _ensure_data_dir() -> None:
    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _candidate_paths() -> Iterable[Path]:
    return [
        DATASET_PATH,
        DEFAULT_DATA_DIR / "train.csv",
        DEFAULT_DATA_DIR / "mobile_train.csv",
        DEFAULT_DATA_DIR / "mobile_price_train.csv",
        DEFAULT_DATA_DIR / "dataset.csv",
    ]


def _download_dataset_if_needed() -> Path | None:
    try:
        import kagglehub
    except Exception:
        return None

    try:
        download_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))
    except Exception:
        return None
    if download_dir.exists():
        return download_dir
    return None


def _find_dataset_in_directory(directory: Path) -> Path | None:
    csv_files = sorted(directory.rglob("*.csv"))
    for csv_file in csv_files:
        if csv_file.name.lower() == "train.csv":
            return csv_file
    for csv_file in csv_files:
        try:
            sample = pd.read_csv(csv_file, nrows=5)
        except Exception:
            continue
        if MOBILE_PRICE_CLASS_COLUMN in sample.columns:
            return csv_file
    return None


def _download_public_fallback(destination: Path) -> Path | None:
    try:
        df = pd.read_csv(PUBLIC_DATASET_FALLBACK_URL)
    except Exception:
        return None
    if MOBILE_PRICE_CLASS_COLUMN not in df.columns:
        return None
    df.to_csv(destination, index=False)
    return destination


def resolve_dataset_path() -> Path:
    for candidate in _candidate_paths():
        if candidate.exists():
            return candidate

    _ensure_data_dir()
    download_dir = _download_dataset_if_needed()
    if download_dir is not None:
        dataset_path = _find_dataset_in_directory(download_dir)
        if dataset_path is not None:
            return dataset_path

    fallback_path = DEFAULT_DATA_DIR / "train.csv"
    mirrored_dataset = _download_public_fallback(fallback_path)
    if mirrored_dataset is not None:
        return mirrored_dataset

    raise FileNotFoundError(
        "Mobile price dataset not found. Download 'dewangmoghe/mobile-phone-price-prediction' "
        "into mlops_project/data or configure internet access/Kaggle credentials so the loader can fetch it."
    )


def load_dataset(dataset_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(dataset_path) if dataset_path else resolve_dataset_path()
    df = pd.read_csv(path)
    return df.drop_duplicates().reset_index(drop=True)


def infer_target_column(df: pd.DataFrame) -> str:
    for column in PRICE_TARGET_CANDIDATES:
        if column in df.columns:
            return column
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_columns:
        raise ValueError("No numeric columns found to infer the price target.")
    return numeric_columns[-1]


def infer_brand_column(df: pd.DataFrame) -> str | None:
    for column in BRAND_CANDIDATES:
        if column in df.columns:
            return column
    return None
