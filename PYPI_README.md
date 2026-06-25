# edRFT: Deep Random Vector Functional Link Transformer Network

A Python package for Deep Random Vector Functional Link Transformer Network models with multiple output layers for significant wave height forecasting.

## Project description

edRFT provides research-grade implementations of Random Vector Functional Link based models for regression and time-series forecasting, with a focus on significant wave height forecasting.

The package includes:

- RVFL-based forecasting utilities
- Deep RVFL Transformer style model components
- Multiple-output-layer forecasting workflow
- Significant wave height forecasting examples
- Hyperparameter tuning utilities
- Evaluation metrics for forecasting experiments

## Key Features

- RVFL-based forecasting models
- Deep architecture with multiple output layers
- Transformer-inspired forecasting workflow
- Support for multivariate wave forecasting data
- Train, validation, and test splitting utilities
- Metrics including RMSE, MAPE, and MASE
- Lightweight package API under edrft

## Installation

Using pip from PyPI:

    pip install edrft

Installing from GitHub:

    pip install git+https://github.com/statsdl/edRFT.git

Installing locally:

    git clone https://github.com/statsdl/edRFT.git
    cd edRFT
    pip install -e .

For development:

    pip install -e ".[dev]"

## Requirements

- Python >= 3.10
- NumPy
- Pandas
- SciPy
- scikit-learn

Optional dependencies may be used for plotting, tuning, and extended examples.

## Quick Start

Example usage:

    import numpy as np
    from edrft import RFTRegressor

    X = np.random.randn(200, 5)
    y = X[:, 0] * 0.5 + X[:, 1] * -0.2 + np.random.randn(200) * 0.05

    model = RFTRegressor(random_state=42)
    model.fit(X, y)

    pred = model.predict(X[:10])
    print(pred)

## Wave Forecasting Example

The package includes a workflow for significant wave height forecasting using meteorological and oceanographic variables.

Typical inputs include:

- Wind direction
- Wind speed
- Gust speed
- Average wave period
- Significant wave height

Run the example from the repository with:

    python examples/run_wave_forecasting.py

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

    @article{bhambu2025deep,
      title={Deep random vector functional link transformer network with multiple output layers for significant wave height forecasting},
      author={Bhambu, Aryan and Gao, Ruobin and Suganthan, Ponnuthurai Nagaratnam and Selvaraju, Natarajan},
      journal={Applied Soft Computing},
      pages={114136},
      year={2025},
      publisher={Elsevier}
    }

## Reference

Bhambu, A., Gao, R., Suganthan, P. N., and Selvaraju, N. (2025). Deep random vector functional link transformer network with multiple output layers for significant wave height forecasting. Applied Soft Computing, 114136.
