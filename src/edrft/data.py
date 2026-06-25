from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def make_forecasting_frame(
    series: Iterable[float] | np.ndarray,
    order: int = 48,
    horizon: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a univariate or multivariate sequence into lagged samples."""

    values = np.asarray(series, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2:
        raise ValueError("series must be a 1D or 2D array.")
    if order <= 0 or horizon <= 0:
        raise ValueError("order and horizon must be positive.")
    n_samples = values.shape[0] - order - horizon + 1
    if n_samples <= 0:
        raise ValueError("series is too short for the requested order and horizon.")
    X = np.zeros((n_samples, values.shape[1] * order), dtype=np.float32)
    y = np.zeros((n_samples, values.shape[1]), dtype=np.float32)
    for i in range(n_samples):
        X[i] = values[i : i + order].ravel()
        y[i] = values[i + order + horizon - 1]
    return X, y.ravel() if y.shape[1] == 1 else y


def chronological_split(n_samples: int, validation_fraction: float = 0.1, test_fraction: float = 0.2):
    """Return train, validation, full-train, and test indexes in time order."""

    test_len = int(test_fraction * n_samples)
    val_len = int(validation_fraction * n_samples)
    train_len = n_samples - val_len - test_len
    if train_len <= 0:
        raise ValueError("Not enough samples for the requested split.")
    train = np.arange(train_len)
    val = np.arange(train_len, train_len + val_len)
    full_train = np.arange(train_len + val_len)
    test = np.arange(train_len + val_len, n_samples)
    return train, val, full_train, test


def load_ndbc_wave_file(path: str | Path, features: list[str] | None = None) -> pd.DataFrame:
    """Load an NDBC wave-height text file and clean sentinel missing values."""

    import pandas as pd

    features = features or ["WDIR", "WSPD", "GST", "APD", "WVHT"]
    frame = pd.read_csv(path, sep=r"\s+", compression="infer")
    frame = frame[features].replace(["99.0", "99.00", 99.0, 99.00], np.nan)
    return frame.ffill().bfill().astype(float)
