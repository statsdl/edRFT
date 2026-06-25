# Project description

Feature - Significant Wave Height Forecasting GitHub last commit GitHub issues GitHub stars Python Version License

A rich documentation is available in the GitHub repository.

# edRFT: Deep Random Vector Functional Link Transformer Network

edRFT is a Python package for significant wave height forecasting using shallow and ensemble deep Random Vector Functional Link Transformer Network models with multiple output layers.

The package is developed for nonlinear time-series forecasting, ocean wave prediction, and regression problems where fast training, strong representation learning, and reliable forecast generation are important.

edRFT combines three useful ideas: randomized neural feature mapping, transformer-inspired feature extraction, and efficient regularized output-layer learning. This allows the model to capture nonlinear relationships in wave data without requiring expensive gradient-based training of all hidden parameters.

The package provides two primary implementations:

RFTRegressor (Random Vector Functional Link Transformer Regressor): A randomized transformer-based forecasting model that converts input variables into nonlinear hidden representations and learns the output layer efficiently using regularized regression.

EDRFTRegressor (Ensemble Deep Random Vector Functional Link Transformer Regressor): A deeper ensemble model that stacks randomized transformer layers and uses multiple output layers. Different layers can capture different levels of information, and their predictions are combined to improve forecasting stability.

Both models are suitable for significant wave height forecasting because wave dynamics are affected by nonlinear interactions among wind direction, wind speed, gust speed, wave period, and previous wave-height observations.

# Key Features

RFT and edRFT Models: Provides both a shallow randomized transformer model and a deeper ensemble version.

Multiple Output Layers: edRFT learns layer-wise output readouts, allowing different hidden depths to contribute to the final forecast.

Transformer-Inspired Feature Interaction: Randomization-based transformer-based mappings help capture nonlinear relationships among input variables.

Efficient Training: Uses randomized hidden representations with regularized output-layer learning, making repeated experiments and tuning practical.

Wave Forecasting: Designed for buoy-based significant wave height forecasting using meteorological and oceanographic variables.

Forecasting Utilities: Includes lag-window creation, scaling, chronological splitting, evaluation, and experiment helpers.

Hyperparameter Tuning: Supports Hyperopt/TPE-based search for reproducible model selection.

# Installation

Downloading Locally and Installing

    git clone https://github.com/statsdl/edRFT.git
    cd edRFT

Install dependencies:

    pip install -r requirements.txt

Install the package:

    pip install -e .

Using pip install from GitHub

    pip install git+https://github.com/statsdl/edRFT.git

Using pip install from PyPI

    pip install edrft

Development installation

    pip install -e ".[dev]"

# Usage

## 1. RFTRegressor

Example

    import numpy as np
    from edrft import RFTRegressor

    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(150, 6))
    y_train = X_train[:, 0] - 0.3 * X_train[:, 1] + np.sin(X_train[:, 2])

    model = RFTRegressor(n_hidden=64, random_state=0)
    model.fit(X_train, y_train)

    X_test = rng.normal(size=(20, 6))
    y_pred = model.predict(X_test)
    print("Predictions:", y_pred)

## 2. EDRFTRegressor

Example

    import numpy as np
    from edrft import EDRFTRegressor, make_forecasting_frame

    series = np.sin(np.linspace(0, 16, 240))

    X, y = make_forecasting_frame(series, order=4, horizon=1)

    X_train, y_train = X[:180], y[:180]
    X_test = X[180:]

    model = EDRFTRegressor(
        n_layers=3,
        n_hidden=32,
        regularization=1e-3,
        random_state=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Forecasts:", y_pred[:5])

# Wave Forecasting Example

The repository includes a command-line workflow for buoy-based significant wave height forecasting.

Run the example:

    python examples/run_wave_forecasting.py \
        --data-dir wave \
        --stations 46001h \
        --years 2017 \
        --look-back 48 \
        --horizon 4 \
        --layers 10 \
        --max-evals 100

The workflow reports RMSE, MAPE, MASE, timing information, and selected hyperparameters.

# API Reference

## RFTRegressor

A randomized transformer-based regressor for regression and forecasting tasks.

Parameters:

n_hidden (int): Number of hidden randomized features.

regularization (float): Ridge regularization parameter.

random_state (int): Random seed for reproducibility.

Methods:

fit(X, y): Fits the model.

predict(X): Predicts output values.

## EDRFTRegressor

A deep ensemble randomized transformer model with multiple output layers.

Parameters:

n_layers (int): Number of stacked randomized transformer layers.

n_hidden (int): Number of hidden features per layer.

regularization (float): Ridge regularization parameter.

aggregation (str): Strategy for combining layer-wise predictions.

random_state (int): Random seed for reproducibility.

Methods:

fit(X, y): Fits the ensemble model.

predict(X): Generates final forecasts.

# Dataset Details

Dataset: NDBC-style significant wave height data.

Typical input variables:

WDIR: Wind direction.

WSPD: Wind speed.

GST: Gust speed.

APD: Average wave period.

WVHT: Significant wave height.

# License

This project is licensed under the MIT License. See the LICENSE file for details.

# Citation

If you use this package in your research, please cite:

    @article{bhambu2025deep,
      title={Deep random vector functional link transformer network with multiple output layers for significant wave height forecasting},
      author={Bhambu, Aryan and Gao, Ruobin and Suganthan, Ponnuthurai Nagaratnam and Selvaraju, Natarajan},
      journal={Applied Soft Computing},
      pages={114136},
      year={2025},
      publisher={Elsevier}
    }
