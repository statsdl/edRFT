import numpy as np
from hyperopt import hp

from edrft.tuning import layerwise_tune_edrft, tune_rft


def test_tune_rft_hyperopt_path():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(36, 3))
    y = X[:, 0] + 0.1 * X[:, 1]

    result = tune_rft(
        X,
        y,
        space={
            "n_hidden": hp.choice("n_hidden", [4, 8]),
            "regularization": hp.choice("regularization", [1e-3]),
            "input_scale": hp.choice("input_scale", [0.2]),
            "transformer_layers": hp.choice("transformer_layers", [1]),
            "num_heads": hp.choice("num_heads", [1]),
            "dropout": hp.choice("dropout", [0.0]),
            "random_state": 2,
        },
        validation_fraction=0.25,
        max_evals=2,
        random_state=2,
    )

    assert len(result.history) == 2
    assert "n_hidden" in result.best_params


def test_layerwise_tune_edrft_hyperopt_path():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 2))
    y = X[:, 0] - X[:, 1]

    result = layerwise_tune_edrft(
        X,
        y,
        n_layers=2,
        layer_space={
            "n_hidden": hp.choice("layer_n_hidden", [4, 8]),
            "regularization": hp.choice("layer_regularization", [1e-3]),
            "input_scale": hp.choice("layer_input_scale", [0.2]),
            "transformer_layers": hp.choice("layer_transformer_layers", [1]),
            "num_heads": hp.choice("layer_num_heads", [1]),
            "dropout": hp.choice("layer_dropout", [0.0]),
        },
        validation_fraction=0.25,
        max_evals=2,
        random_state=3,
    )

    assert len(result.best_params["layer_params"]) == 2
    assert len(result.history) == 4
