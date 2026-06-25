from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

from .metrics import root_mean_squared_error
from .models import EDRFTRegressor, RFTLayerParams, RFTRegressor

Scorer = Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True)
class TuningResult:
    model: Any
    best_params: dict[str, Any]
    best_score: float
    history: list[dict[str, Any]]


def default_rft_space(seed: int = 0) -> dict[str, Any]:
    """Hyperopt space matching the RFT/default edRFT search ranges."""

    return {
        "n_hidden": hp.quniform("n_hidden", 32, 1024, 32),
        "regularization": hp.uniform("regularization", 0, 1),
        "input_scale": hp.uniform("input_scale", 0, 1),
        "transformer_layers": hp.choice("transformer_layers", [1, 2, 3, 4, 5]),
        "num_heads": hp.choice("num_heads", [2, 4, 6, 8]),
        "dropout": hp.uniform("dropout", 0, 0.5),
        "random_state": seed,
    }


def default_edrft_layer_space() -> dict[str, Any]:
    return {
        "n_hidden": hp.quniform("n_hidden", 32, 1024, 32),
        "regularization": hp.uniform("regularization", 0, 1),
        "input_scale": hp.uniform("input_scale", 0, 1),
        "transformer_layers": hp.choice("transformer_layers", [1, 2, 3, 4, 5]),
        "num_heads": hp.choice("num_heads", [2, 4, 6, 8]),
        "dropout": hp.uniform("dropout", 0, 0.5),
    }


def tune_rft(
    X,
    y,
    space: dict[str, Any] | None = None,
    scorer: Scorer = root_mean_squared_error,
    validation_fraction: float = 0.2,
    max_evals: int = 100,
    random_state: int = 0,
    refit: bool = True,
) -> TuningResult:
    return _tune(
        RFTRegressor,
        X,
        y,
        default_rft_space(random_state) if space is None else space,
        scorer,
        validation_fraction,
        max_evals,
        random_state,
        refit,
    )


def layerwise_tune_edrft(
    X,
    y,
    n_layers: int = 10,
    layer_space: dict[str, Any] | None = None,
    scorer: Scorer = root_mean_squared_error,
    validation_fraction: float = 0.2,
    max_evals: int = 100,
    random_state: int = 0,
    refit: bool = True,
    fixed_params: dict[str, Any] | None = None,
) -> TuningResult:
    if n_layers <= 0:
        raise ValueError("n_layers must be positive.")
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    X_train, X_val, y_train, y_val = _split(X_arr, y_arr, validation_fraction)
    selected: list[RFTLayerParams] = []
    history = []
    fixed_params = dict(fixed_params or {})
    space = default_edrft_layer_space() if layer_space is None else layer_space

    for layer_index in range(n_layers):
        layer_history = []

        def objective(params):
            clean = _clean_params(params)
            if layer_index > 0:
                clean["n_hidden"] = selected[0].n_hidden
            candidate = selected + [RFTLayerParams(**clean)]
            model = EDRFTRegressor(layer_params=candidate, random_state=random_state, **fixed_params).fit(
                X_train, y_train
            )
            score = float(scorer(y_val, model.predict(X_val)))
            record = {"layer": layer_index + 1, "score": score, **clean}
            layer_history.append(record)
            history.append(record)
            return {"loss": score, "status": STATUS_OK}

        fmin(
            objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=Trials(),
            rstate=np.random.default_rng(random_state + layer_index),
            show_progressbar=False,
        )
        best = min(layer_history, key=lambda item: item["score"])
        selected.append(
            RFTLayerParams(
                n_hidden=int(best["n_hidden"]),
                regularization=float(best["regularization"]),
                input_scale=float(best["input_scale"]),
                transformer_layers=int(best["transformer_layers"]),
                num_heads=int(best["num_heads"]),
                dropout=float(best["dropout"]),
            )
        )

    final_X, final_y = (X_arr, y_arr) if refit else (X_train, y_train)
    best_params = {"layer_params": [param.__dict__ for param in selected], **fixed_params}
    return TuningResult(
        model=EDRFTRegressor(layer_params=selected, random_state=random_state, **fixed_params).fit(final_X, final_y),
        best_params=best_params,
        best_score=min(item["score"] for item in history if item["layer"] == n_layers),
        history=history,
    )


def _tune(estimator_cls, X, y, space, scorer, validation_fraction, max_evals, random_state, refit):
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    X_train, X_val, y_train, y_val = _split(X_arr, y_arr, validation_fraction)
    history = []

    def objective(params):
        clean = _clean_params(params)
        model = estimator_cls(**clean).fit(X_train, y_train)
        score = float(scorer(y_val, model.predict(X_val)))
        history.append({"score": score, **clean})
        return {"loss": score, "status": STATUS_OK}

    fmin(
        objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=Trials(),
        rstate=np.random.default_rng(random_state),
        show_progressbar=False,
    )
    best = min(history, key=lambda item: item["score"])
    best_params = {key: value for key, value in best.items() if key != "score"}
    final_X, final_y = (X_arr, y_arr) if refit else (X_train, y_train)
    return TuningResult(
        model=estimator_cls(**best_params).fit(final_X, final_y),
        best_params=best_params,
        best_score=float(best["score"]),
        history=history,
    )


def _split(X, y, validation_fraction):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    n_val = max(1, int(round(len(X) * validation_fraction)))
    return X[:-n_val], X[-n_val:], y[:-n_val], y[-n_val:]


def _clean_params(params):
    clean = dict(params)
    for key in ("n_hidden", "transformer_layers", "num_heads", "random_state"):
        if key in clean:
            clean[key] = int(clean[key])
    if "n_hidden" in clean:
        clean["n_hidden"] = max(1, clean["n_hidden"])
    for key in ("regularization", "input_scale", "dropout"):
        if key in clean:
            clean[key] = float(clean[key])
    return clean
