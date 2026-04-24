from __future__ import annotations

import math

from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_classification(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
