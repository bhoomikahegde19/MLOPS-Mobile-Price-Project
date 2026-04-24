from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import RANDOM_STATE, TEST_SIZE


@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    target_column: str


def prepare_features(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    clean_df = df.drop_duplicates().reset_index(drop=True).copy()
    X = clean_df.drop(columns=[target_column])
    y = clean_df[target_column]
    return X, y


def split_data(df: pd.DataFrame, target_column: str) -> SplitData:
    X, y = prepare_features(df, target_column)
    stratify = None
    if y.nunique(dropna=True) <= 10:
        stratify = y
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    return SplitData(X_train, X_test, y_train, y_test, target_column)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )


def transform_with_preprocessor(preprocessor: ColumnTransformer, X: pd.DataFrame) -> Any:
    transformed = preprocessor.transform(X)
    if hasattr(transformed, "toarray"):
        return transformed.toarray()
    return transformed
