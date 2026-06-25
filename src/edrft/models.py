from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn as nn

Aggregation = Literal["median", "mean"]


@dataclass(frozen=True)
class RFTLayerParams:
    """Hyperparameters for one RFT hidden layer."""

    n_hidden: int = 64
    regularization: float = 1e-3
    input_scale: float = 0.1
    transformer_layers: int = 1
    num_heads: int = 1
    dropout: float = 0.0


def _as_2d(values: np.ndarray | Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def _ridge_solve(design: np.ndarray, target: np.ndarray, regularization: float) -> np.ndarray:
    penalty = float(regularization) * np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    left = design.T @ design + penalty
    right = design.T @ target
    try:
        return np.linalg.solve(left, right)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(left) @ right


class _RandomTransformerBlock(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        input_scale: float,
        transformer_layers: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        heads = int(num_heads) if n_features % int(num_heads) == 0 else 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_features,
            nhead=max(1, heads),
            dim_feedforward=int(n_hidden),
            dropout=float(dropout),
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(transformer_layers))
        self.projection = nn.Linear(n_features, int(n_hidden))
        self.activation = nn.Tanh()
        self.input_scale = float(input_scale)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for parameter in self.encoder.parameters():
            if parameter.ndim > 1:
                nn.init.uniform_(parameter, -self.input_scale, self.input_scale)
            else:
                nn.init.zeros_(parameter)
        nn.init.uniform_(self.projection.weight, -self.input_scale, self.input_scale)
        nn.init.uniform_(self.projection.bias, -self.input_scale, self.input_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x.unsqueeze(1)).squeeze(1)
        return self.activation(self.projection(encoded))


class RFTRegressor:
    """Random Vector Functional Link Transformer regressor.

    Hidden transformer weights are randomly initialized and fixed. Only the
    output readout is solved with ridge regression.
    """

    def __init__(
        self,
        n_hidden: int = 64,
        regularization: float = 1e-3,
        input_scale: float = 0.1,
        transformer_layers: int = 1,
        num_heads: int = 1,
        dropout: float = 0.0,
        random_state: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.n_hidden = n_hidden
        self.regularization = regularization
        self.input_scale = input_scale
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.random_state = random_state
        self.device = device

    def fit(self, X: np.ndarray | Iterable[float], y: np.ndarray | Iterable[float]) -> "RFTRegressor":
        X_arr = _as_2d(X, "X")
        y_arr = _as_2d(y, "y").astype(np.float64)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        torch.manual_seed(0 if self.random_state is None else int(self.random_state))
        self.block_ = _RandomTransformerBlock(
            X_arr.shape[1],
            int(self.n_hidden),
            float(self.input_scale),
            int(self.transformer_layers),
            int(self.num_heads),
            float(self.dropout),
        ).to(self.device)
        self.block_.eval()
        hidden = self._hidden(X_arr)
        design = self._design(X_arr, hidden)
        self.coef_ = _ridge_solve(design, y_arr, float(self.regularization))
        self.n_features_in_ = X_arr.shape[1]
        self.n_outputs_ = y_arr.shape[1]
        return self

    def predict(self, X: np.ndarray | Iterable[float]) -> np.ndarray:
        self._check_fitted()
        X_arr = _as_2d(X, "X")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_arr.shape[1]}.")
        pred = self._design(X_arr, self._hidden(X_arr)) @ self.coef_
        return pred.ravel() if self.n_outputs_ == 1 else pred

    def _hidden(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
            return self.block_(tensor).cpu().numpy().astype(np.float64)

    @staticmethod
    def _design(X: np.ndarray, hidden: np.ndarray) -> np.ndarray:
        return np.hstack([hidden, X.astype(np.float64), np.ones((X.shape[0], 1))])

    def _check_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise RuntimeError("The model must be fitted before prediction.")


class EDRFTRegressor:
    """Ensemble deep RFT regressor with one ridge readout per layer."""

    def __init__(
        self,
        n_layers: int = 3,
        n_hidden: int = 64,
        regularization: float = 1e-3,
        input_scale: float = 0.1,
        transformer_layers: int = 1,
        num_heads: int = 1,
        dropout: float = 0.0,
        aggregation: Aggregation = "median",
        random_state: int | None = None,
        device: str = "cpu",
        layer_params: list[RFTLayerParams | dict] | None = None,
    ) -> None:
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.regularization = regularization
        self.input_scale = input_scale
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.aggregation = aggregation
        self.random_state = random_state
        self.device = device
        self.layer_params = layer_params

    def fit(self, X: np.ndarray | Iterable[float], y: np.ndarray | Iterable[float]) -> "EDRFTRegressor":
        X_arr = _as_2d(X, "X")
        y_arr = _as_2d(y, "y").astype(np.float64)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")

        torch.manual_seed(0 if self.random_state is None else int(self.random_state))
        state = X_arr
        self.layers_ = []
        for params in self._resolved_layer_params():
            block = _RandomTransformerBlock(
                state.shape[1],
                params.n_hidden,
                params.input_scale,
                params.transformer_layers,
                params.num_heads,
                params.dropout,
            ).to(self.device)
            block.eval()
            with torch.no_grad():
                hidden = block(torch.as_tensor(state, dtype=torch.float32, device=self.device))
            hidden_np = hidden.cpu().numpy().astype(np.float64)
            design = np.hstack([hidden_np, X_arr.astype(np.float64), np.ones((X_arr.shape[0], 1))])
            coef = _ridge_solve(design, y_arr, params.regularization)
            self.layers_.append({"block": block, "coef": coef, "params": params})
            state = hidden_np.astype(np.float32)

        self.n_features_in_ = X_arr.shape[1]
        self.n_outputs_ = y_arr.shape[1]
        return self

    def predict(self, X: np.ndarray | Iterable[float], return_layers: bool = False) -> np.ndarray:
        self._check_fitted()
        X_arr = _as_2d(X, "X")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X_arr.shape[1]}.")
        state = X_arr
        outputs = []
        for layer in self.layers_:
            with torch.no_grad():
                hidden = layer["block"](torch.as_tensor(state, dtype=torch.float32, device=self.device))
            hidden_np = hidden.cpu().numpy().astype(np.float64)
            design = np.hstack([hidden_np, X_arr.astype(np.float64), np.ones((X_arr.shape[0], 1))])
            outputs.append(design @ layer["coef"])
            state = hidden_np.astype(np.float32)

        stacked = np.stack(outputs, axis=0)
        if return_layers:
            result = np.moveaxis(stacked, 0, 1)
        elif self.aggregation == "mean":
            result = np.mean(stacked, axis=0)
        elif self.aggregation == "median":
            result = np.median(stacked, axis=0)
        else:
            raise ValueError("aggregation must be 'mean' or 'median'.")
        return result.ravel() if self.n_outputs_ == 1 and result.ndim == 2 else result

    def _resolved_layer_params(self) -> list[RFTLayerParams]:
        if self.layer_params is None:
            return [
                RFTLayerParams(
                    self.n_hidden,
                    self.regularization,
                    self.input_scale,
                    self.transformer_layers,
                    self.num_heads,
                    self.dropout,
                )
                for _ in range(int(self.n_layers))
            ]
        params = []
        for item in self.layer_params:
            params.append(item if isinstance(item, RFTLayerParams) else RFTLayerParams(**item))
        if not params:
            raise ValueError("layer_params cannot be empty.")
        return params

    def _check_fitted(self) -> None:
        if not hasattr(self, "layers_"):
            raise RuntimeError("The model must be fitted before prediction.")
