import numpy as np

from edrft import EDRFTRegressor, RFTRegressor, make_forecasting_frame


def test_rft_predicts_expected_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(32, 4))
    y = X[:, 0] - 0.2 * X[:, 1]

    model = RFTRegressor(n_hidden=8, transformer_layers=1, num_heads=1, random_state=0).fit(X, y)
    pred = model.predict(X[:6])

    assert pred.shape == (6,)
    assert np.all(np.isfinite(pred))


def test_edrft_layer_outputs():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(36, 3))
    y = np.sin(X[:, 0]) + X[:, 2]

    model = EDRFTRegressor(n_layers=2, n_hidden=8, random_state=1).fit(X, y)
    pred = model.predict(X[:5])
    layers = model.predict(X[:5], return_layers=True)

    assert pred.shape == (5,)
    assert layers.shape == (5, 2, 1)


def test_forecasting_frame():
    series = np.sin(np.linspace(0, 4, 60))
    X, y = make_forecasting_frame(series, order=4, horizon=1)

    assert X.shape == (56, 4)
    assert y.shape == (56,)
