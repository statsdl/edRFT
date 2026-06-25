from __future__ import annotations

import numpy as np


def root_mean_squared_error(y_true, y_pred) -> float:
    truth = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def mean_absolute_percentage_error(y_true, y_pred, epsilon: float = 1e-8) -> float:
    truth = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    denom = np.maximum(np.abs(truth), epsilon)
    return float(np.mean(np.abs((truth - pred) / denom)))


def mean_absolute_scaled_error(y_true, y_pred, history, seasonality: int = 1) -> float:
    truth = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    hist = np.asarray(history, dtype=float).ravel()
    scale = np.mean(np.abs(hist[seasonality:] - hist[:-seasonality]))
    if scale == 0:
        return float("inf")
    return float(np.mean(np.abs(truth - pred)) / scale)
