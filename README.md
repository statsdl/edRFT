# edRFT: Deep Random Vector Functional Link Transformer Network

<p align="center">
  <strong>🌊 Fast randomized transformer models for significant wave-height forecasting 🌊</strong>
</p>

<p align="center">
  <a href="https://github.com/statsdl/edRFT/actions/workflows/test.yml"><img alt="Tests" src="https://github.com/statsdl/edRFT/actions/workflows/test.yml/badge.svg"></a>
  <a href="https://pypi.org/project/edrft/"><img alt="PyPI" src="https://img.shields.io/pypi/v/edrft.svg"></a>
  <a href="https://pypi.org/project/edrft/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/edrft.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

**edRFT** provides research-grade implementations of Random Vector Functional Link Transformer models for regression and significant wave-height forecasting. The package includes a shallow **RFT** model, an ensemble deep **edRFT** model with multiple output layers, forecasting utilities, error metrics, and Hyperopt-based tuning helpers.

The method combines randomized transformer feature mappings with closed-form ridge readouts, giving a practical balance between deep representation learning and efficient training.

## Contents

- [Use Cases](#use-cases)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Wave Forecasting Demo](#wave-forecasting-demo)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Mathematical Overview](#mathematical-overview)
- [Contributing](#contributing)
- [Citation](#citation)

## Use Cases

### 🌊 Ocean and Coastal Forecasting

- Significant wave-height forecasting
- Buoy-based marine condition prediction
- Short-horizon ocean-state modelling

### 📈 Time-Series Forecasting

- Regression with lagged input windows
- Multi-step forecasting experiments
- Nonlinear temporal pattern learning

### 🔬 Research Prototyping

- Randomized neural network baselines
- Transformer-enhanced RVFL experiments
- Fast benchmarking with ridge-regression readouts

## Key Features

### 🎯 Methodology

- **Random Vector Functional Link Transformer (RFT)** model
- **Ensemble deep RFT (edRFT)** with one output readout per hidden layer
- Randomized transformer blocks with fixed hidden weights
- Closed-form ridge-regression readouts
- Forecasting-frame creation utilities
- RMSE, MAPE, and MASE metrics

### ⚡ Practical Workflow

- Lightweight core API
- Optional wave-forecasting utilities
- Hyperopt/TPE tuning helpers
- Reproducible `random_state` support
- GitHub Actions test and PyPI release workflows

## Why edRFT?

| Feature | Classical RVFL | Transformer Models | edRFT |
|---|---:|---:|---:|
| Randomized hidden mapping | ✅ | ❌ | ✅ |
| Transformer-based feature interaction | ❌ | ✅ | ✅ |
| Closed-form output-layer training | ✅ | ❌ | ✅ |
| Multiple output layers | ❌ | ⚠️ Architecture-dependent | ✅ |
| Fast regression-style fitting | ✅ | ❌ | ✅ |
| Time-series forecasting utilities | ⚠️ Manual | ⚠️ Manual | ✅ |

## Installation

Install the core package from PyPI:

```bash
pip install edrft
```

Install optional wave-forecasting dependencies:

```bash
pip install "edrft[wave]"
```

Install from source:

```bash
git clone https://github.com/statsdl/edRFT.git
cd edRFT
pip install -e ".[dev]"
pytest
```

## Requirements

Core requirements:

- Python >= 3.10
- NumPy
- PyTorch

Optional dependencies:

- `hyperopt` for tuning
- `pandas` for wave data loading
- `build`, `twine`, and `pytest` for development and release workflows

## Quick Start

```python
import numpy as np
from edrft import EDRFTRegressor, make_forecasting_frame

# Create a simple univariate forecasting dataset
series = np.sin(np.linspace(0, 16, 240))
X, y = make_forecasting_frame(series, order=4, horizon=1)

# Train/test split
X_train, y_train = X[:180], y[:180]
X_test = X[180:]

# Fit edRFT
model = EDRFTRegressor(
    n_layers=3,
    n_hidden=32,
    regularization=1e-3,
    random_state=0,
)
model.fit(X_train, y_train)

# Forecast
pred = model.predict(X_test)
print(pred[:5])
```

## RFT Example

```python
import numpy as np
from edrft import RFTRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 6))
y = X[:, 0] - 0.3 * X[:, 1] + np.sin(X[:, 2])

model = RFTRegressor(n_hidden=64, random_state=0)
model.fit(X[:150], y[:150])
pred = model.predict(X[150:])
```

## Wave Forecasting Demo

The repository includes a command-line demo for buoy-based significant wave-height forecasting:

```bash
python examples/run_wave_forecasting.py   --data-dir wave   --stations 46001h   --years 2017   --seeds 0   --look-back 48   --horizon 4   --layers 10   --max-evals 100
```

The demo prints RMSE, MAPE, MASE, tuning time, training time, testing time, and selected hyperparameters.

## Hyperparameter Tuning

```python
from edrft.tuning import layerwise_tune_edrft

result = layerwise_tune_edrft(
    X,
    y,
    n_layers=10,
    validation_fraction=0.125,
    max_evals=100,
    random_state=0,
)

print(result.best_params)
print(result.history[-1])
```

## Mathematical Overview

For an input vector `x`, an RFT hidden layer creates a randomized transformer feature map:

```text
h = phi(T(x; W_T), W_p)
```

where `T` is a transformer encoder with fixed randomized parameters, `W_p` is a randomized projection, and `phi` is a nonlinear activation. The output is learned with ridge regression:

```text
beta = argmin ||H beta - y||² + lambda ||beta||²
```

The edRFT model stacks multiple randomized transformer layers and learns a separate output readout at each layer. Final prediction is obtained by aggregating the layer-wise predictions, for example by median or mean aggregation.

## Repository Layout

```text
src/edrft/        Supported package API
examples/         Usage examples
tests/            Unit tests
.github/          CI and PyPI publishing workflows
wave/             Sample wave-height data files
```

The supported public API lives in `src/edrft`. Use the examples and package modules for new work.

## Getting Help

- Open an issue: https://github.com/statsdl/edRFT/issues
- Check examples in `examples/`
- Use the package API from `src/edrft`

## Contributing

Contributions are welcome. Please open an issue or pull request for bug fixes, documentation improvements, tests, or examples.

Before submitting changes, run:

```bash
pip install -e ".[dev]"
pytest
```

## License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

## Citation

If you use edRFT in your work, please cite:

```bibtex
@article{bhambu2025deep,
  title={Deep random vector functional link transformer network with multiple output layers for significant wave height forecasting},
  author={Bhambu, Aryan and Gao, Ruobin and Suganthan, Ponnuthurai Nagaratnam and Selvaraju, Natarajan},
  journal={Applied Soft Computing},
  pages={114136},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgments

edRFT builds on concepts from randomized neural networks, RVFL models, transformer-based representation learning, and significant wave-height forecasting research.

If this package supports your work, please consider starring the repository.
