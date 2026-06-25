from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np

from .data import chronological_split, load_ndbc_wave_file
from .metrics import mean_absolute_percentage_error, mean_absolute_scaled_error, root_mean_squared_error
from .models import EDRFTRegressor, RFTRegressor
from .tuning import layerwise_tune_edrft, tune_rft


@dataclass(frozen=True)
class WaveRunResult:
    station: str
    year: str
    seed: int
    model: str
    rmse: float
    mape: float
    mase: float
    tuning_seconds: float
    training_seconds: float
    testing_seconds: float
    best_params: dict


def prepare_wave_supervised(
    path: str | Path,
    look_back: int = 48,
    horizon: int = 4,
    features: list[str] | None = None,
    order: int | None = None,
):
    """Prepare NDBC wave data using wave forecasting lag and horizon settings."""

    if order is not None:
        look_back = order
    if look_back <= 0 or horizon <= 0:
        raise ValueError("look_back and horizon must be positive.")
    frame = load_ndbc_wave_file(path, features=features)
    values = frame.to_numpy(dtype=float)[1:]
    target = frame["WVHT"].to_numpy(dtype=float)[1:]
    n_samples = values.shape[0] - look_back - horizon + 1
    if n_samples <= 0:
        raise ValueError("series is too short for the requested look_back and horizon.")
    X = np.zeros((n_samples, values.shape[1] * look_back), dtype=np.float32)
    y = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        X[i] = values[i : i + look_back].ravel()
        y[i] = target[i + look_back + horizon - 1]
    return X, y


def run_wave_experiment(
    data_dir: str | Path = "wave",
    stations: Iterable[str] = ("46001h",),
    years: Iterable[str] = ("2017",),
    seeds: Iterable[int] = (0,),
    look_back: int = 48,
    horizon: int = 4,
    n_layers: int = 10,
    max_evals: int = 100,
    order: int | None = None,
) -> list[WaveRunResult]:
    """Run RFT and edRFT on NDBC files without writing result files."""

    if order is not None:
        look_back = order
    data_dir = Path(data_dir)
    results: list[WaveRunResult] = []
    for year in years:
        for station in stations:
            X, y = prepare_wave_supervised(
                data_dir / f"{station}{year}.txt.gz",
                look_back=look_back,
                horizon=horizon,
            )
            train_idx, val_idx, full_train_idx, test_idx = chronological_split(len(X), 0.1, 0.2)
            scaled = _scaled_splits(X, y, train_idx, val_idx, full_train_idx, test_idx)
            for seed in seeds:
                results.append(
                    _run_rft(
                        station,
                        year,
                        seed,
                        scaled,
                        max_evals=max_evals,
                    )
                )
                results.append(
                    _run_edrft(
                        station,
                        year,
                        seed,
                        scaled,
                        n_layers=n_layers,
                        max_evals=max_evals,
                    )
                )
    return results


def _run_rft(station, year, seed, scaled, max_evals):
    train, val, full_train, test, inverse_y, history = scaled
    tune_X = np.vstack([train[0], val[0]])
    tune_y = np.concatenate([train[1], val[1]])

    start = perf_counter()
    tuned = tune_rft(
        tune_X,
        tune_y,
        validation_fraction=len(val[1]) / len(tune_y),
        max_evals=max_evals,
        random_state=seed,
        refit=False,
    )
    tuning_seconds = perf_counter() - start

    start = perf_counter()
    model = RFTRegressor(**tuned.best_params).fit(*full_train)
    training_seconds = perf_counter() - start

    start = perf_counter()
    pred = model.predict(test[0])
    testing_seconds = perf_counter() - start
    return _result(station, year, seed, "RFT", test[1], pred, inverse_y, history, tuning_seconds, training_seconds, testing_seconds, tuned.best_params)


def _run_edrft(station, year, seed, scaled, n_layers, max_evals):
    train, val, full_train, test, inverse_y, history = scaled
    tune_X = np.vstack([train[0], val[0]])
    tune_y = np.concatenate([train[1], val[1]])

    start = perf_counter()
    tuned = layerwise_tune_edrft(
        tune_X,
        tune_y,
        n_layers=n_layers,
        validation_fraction=len(val[1]) / len(tune_y),
        max_evals=max_evals,
        random_state=seed,
        refit=False,
    )
    tuning_seconds = perf_counter() - start

    start = perf_counter()
    model = EDRFTRegressor(layer_params=tuned.best_params["layer_params"], random_state=seed).fit(*full_train)
    training_seconds = perf_counter() - start

    start = perf_counter()
    pred = model.predict(test[0])
    testing_seconds = perf_counter() - start
    return _result(station, year, seed, "edRFT", test[1], pred, inverse_y, history, tuning_seconds, training_seconds, testing_seconds, tuned.best_params)


def _scaled_splits(X, y, train_idx, val_idx, full_train_idx, test_idx):
    x_min = X[train_idx].min(axis=0)
    x_range = np.maximum(X[train_idx].max(axis=0) - x_min, 1e-12)
    y_min = y[train_idx].min()
    y_range = max(float(y[train_idx].max() - y_min), 1e-12)

    def scale_x(values):
        return 2 * ((values - x_min) / x_range) - 1

    def scale_y(values):
        return 2 * ((values - y_min) / y_range) - 1

    def inverse_y(values):
        return ((np.asarray(values, dtype=float) + 1) / 2) * y_range + y_min

    return (
        (scale_x(X[train_idx]), scale_y(y[train_idx])),
        (scale_x(X[val_idx]), scale_y(y[val_idx])),
        (scale_x(X[full_train_idx]), scale_y(y[full_train_idx])),
        (scale_x(X[test_idx]), scale_y(y[test_idx])),
        inverse_y,
        y[train_idx],
    )


def _result(station, year, seed, model, y_true, y_pred, inverse_y, history, tuning, training, testing, best_params):
    truth = inverse_y(y_true)
    pred = inverse_y(y_pred)
    return WaveRunResult(
        station=station,
        year=year,
        seed=seed,
        model=model,
        rmse=root_mean_squared_error(truth, pred),
        mape=mean_absolute_percentage_error(truth, pred),
        mase=mean_absolute_scaled_error(truth, pred, history),
        tuning_seconds=tuning,
        training_seconds=training,
        testing_seconds=testing,
        best_params=best_params,
    )
