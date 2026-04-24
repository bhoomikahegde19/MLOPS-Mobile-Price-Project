from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.config import RANDOM_STATE, TEST_SIZE
from pipeline.data.preprocess import prepare_features_and_target


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def create_train_test_split(df: pd.DataFrame) -> DatasetSplit:
    X, y = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return DatasetSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
