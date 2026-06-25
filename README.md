# edRFT

`edrft` provides Random Vector Functional Link Transformer models for
significant wave-height forecasting:

- `RFTRegressor`: a shallow randomized transformer encoder with ridge readout.
- `EDRFTRegressor`: an ensemble deep RFT with one output layer per hidden layer.
- Hyperopt/TPE tuning using the default edRFT search ranges.
- NDBC wave forecasting experiment helpers that do not write result artifacts.

The public package uses the model naming: `RFT` and `edRFT`. Older `rft` and
`edrft` script names are retained only under legacy files for traceability.

## Installation

Core install:

```bash
git clone https://github.com/statsdl/edRFT.git
cd edRFT
pip install .
```

Wave experiment dependencies:

```bash
pip install ".[wave]"
```

Development:

```bash
pip install -e ".[dev]"
pytest
```

## Quick Start

```python
import numpy as np
from edrft import EDRFTRegressor, make_forecasting_frame

series = np.sin(np.linspace(0, 16, 240))
X, y = make_forecasting_frame(series, order=4)

model = EDRFTRegressor(n_layers=3, n_hidden=32, random_state=0)
model.fit(X[:180], y[:180])
pred = model.predict(X[180:])
```

## Wave Forecasting Example

```bash
python examples/run_wave_forecasting.py \
  --data-dir wave \
  --stations 46001h \
  --years 2017 \
  --seeds 0 \
  --look-back 48 \
  --horizon 4 \
  --layers 10 \
  --max-evals 100
```

The runner prints metrics to stdout only. It follows the original scripts:

- NDBC features: `WDIR`, `WSPD`, `GST`, `APD`, `WVHT`
- Missing sentinel cleanup
- Default look-back window: 48
- Default forecasting horizon: 4
- Min-max scaling to `[-1, 1]`
- Chronological split: 70% train, 10% validation, 20% test
- Hyperopt/TPE tuning with 100 evaluations by default
- Train+validation final fit
- RMSE, MAPE, MASE, and timing output

## Hyperopt Tuning

```python
from edrft.tuning import layerwise_tune_edrft

result = layerwise_tune_edrft(
    X,
    y,
    n_layers=10,
    validation_fraction=0.1 / 0.8,
    max_evals=100,
    random_state=0,
)
```

## Repository Notes

Supported package code lives in `src/edrft`.

The `legacy/`, `DeepRVFL_/`, `ForecastLib.py`, and old experiment scripts are
retained for traceability. They are not included in the PyPI wheel and are not
the supported package API.

## PyPI Release

The publish workflow uses PyPI Trusted Publishing. Configure the PyPI trusted
publisher with:

- owner: `statsdl`
- repository: `edRFT`
- workflow: `publish.yml`
- environment: `pypi`

## License

MIT

## Reference

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

