from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline.config import DATASET_PATH, DATA_DIR, KAGGLE_DATASET_SLUG, PUBLIC_DATASET_FALLBACK_URL, TARGET_COLUMN


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _download_from_kaggle() -> Path | None:
    try:
        import kagglehub
    except Exception:
        return None

    try:
        download_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET_SLUG))
    except Exception:
        return None

    for csv_file in sorted(download_dir.rglob("*.csv")):
        try:
            sample = pd.read_csv(csv_file, nrows=5)
        except Exception:
            continue
        if TARGET_COLUMN in sample.columns:
            return csv_file
    return None


def _download_public_fallback(destination: Path) -> Path | None:
    try:
        df = pd.read_csv(PUBLIC_DATASET_FALLBACK_URL)
    except Exception:
        return None
    if TARGET_COLUMN not in df.columns:
        return None
    df.to_csv(destination, index=False)
    return destination


def resolve_dataset_path() -> Path:
    if DATASET_PATH.exists():
        return DATASET_PATH

    _ensure_data_dir()
    kaggle_path = _download_from_kaggle()
    if kaggle_path is not None:
        return kaggle_path

    mirrored_path = _download_public_fallback(DATA_DIR / "train.csv")
    if mirrored_path is not None:
        return mirrored_path

    raise FileNotFoundError(
        "Dataset could not be found or downloaded. Place the Kaggle train.csv in mlops_mobile_price/data."
    )


def load_csv() -> pd.DataFrame:
    dataset_path = resolve_dataset_path()
    df = pd.read_csv(dataset_path)
    return df.drop_duplicates().reset_index(drop=True)
