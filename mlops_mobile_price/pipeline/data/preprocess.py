from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.config import FEATURE_COLUMNS, TARGET_COLUMN


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    usable_columns = [column for column in FEATURE_COLUMNS if column in df.columns] + [TARGET_COLUMN]
    clean_df = df[usable_columns].copy()
    return clean_df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)


def build_numeric_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_columns)],
        remainder="drop",
    )


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    clean_df = clean_dataframe(df)
    X = clean_df.drop(columns=[TARGET_COLUMN])
    y = clean_df[TARGET_COLUMN]
    return X, y
