"""Pluggable forecasting models. Each registers itself and implements IModel,
so the selector compares them and the API loads any of them the same way.

Both predict a full `horizon`-length vector (multi-output).
"""
from __future__ import annotations

import numpy as np

from forecasting.core.registry import register_model
from forecasting.interfaces.contracts import IModel


@register_model("xgboost")
class XGBoostForecaster(IModel):
    """Multi-output XGBoost via one regressor per horizon step."""

    def __init__(self, horizon: int = 7, **params):
        self.horizon = horizon
        self.params = params or dict(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
        )
        self.models = []

    def fit(self, X, y):
        from xgboost import XGBRegressor
        y = np.asarray(y)
        self.horizon = y.shape[1]
        self.models = []
        for h in range(self.horizon):
            m = XGBRegressor(n_jobs=-1, **self.params)
            m.fit(X, y[:, h])
            self.models.append(m)
        return self

    def predict(self, X):
        return np.column_stack([m.predict(X) for m in self.models])

    def save(self, path: str):
        import joblib
        joblib.dump({"models": self.models, "horizon": self.horizon}, path)

    @classmethod
    def load(cls, path: str):
        import joblib
        d = joblib.load(path)
        o = cls(horizon=d["horizon"])
        o.models = d["models"]
        return o


@register_model("lstm")
class LSTMForecaster(IModel):
    """Sequence-to-vector LSTM: lookback window -> horizon-length forecast.

    Scales its own target. The window builder hands every model the raw target
    history in channel 0 of X_seq and raw values in y; trees are scale-invariant
    but a network trained on values in the thousands under MSE starts with a
    loss in the tens of millions and never leaves zero. Standardising here — on
    training statistics only, inverted at predict time — keeps the comparison
    against XGBoost fair without changing the pipeline or the interface.
    """

    def __init__(self, horizon: int = 7, units: int = 64, epochs: int = 15,
                 batch_size: int = 64):
        self.horizon = horizon
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self._mu: float | None = None
        self._sigma: float | None = None

    # -- target scaling --
    def _fit_scaler(self, y) -> None:
        y = np.asarray(y, dtype=float)
        self._mu = float(np.nanmean(y))
        sigma = float(np.nanstd(y))
        self._sigma = sigma if sigma > 1e-8 else 1.0

    def _scale(self, a):
        return (np.asarray(a, dtype="float32") - self._mu) / self._sigma

    def _unscale(self, a):
        return np.asarray(a, dtype=float) * self._sigma + self._mu

    def _scale_inputs(self, X):
        """Channel 0 of each window is the target history; the rest are already
        preprocessed features and are left alone."""
        X = np.array(X, dtype="float32", copy=True)
        X[:, :, 0] = self._scale(X[:, :, 0])
        return X

    def _build(self, n_timesteps, n_features):
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
        from tensorflow.keras.models import Sequential
        m = Sequential([
            Input(shape=(n_timesteps, n_features)),
            LSTM(self.units),
            Dropout(0.2),
            Dense(self.units // 2, activation="relu"),
            Dense(self.horizon),
        ])
        m.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return m

    def fit(self, X, y):
        y = np.asarray(y)
        self.horizon = y.shape[1]
        self._fit_scaler(y)
        self.model = self._build(X.shape[1], X.shape[2])
        self.model.fit(
            self._scale_inputs(X), self._scale(y),
            epochs=self.epochs, batch_size=self.batch_size,
            validation_split=0.1, verbose=0,
        )
        return self

    def predict(self, X):
        scaled = self.model.predict(self._scale_inputs(X), verbose=0)
        return self._unscale(scaled)

    @staticmethod
    def _scaler_path(path: str) -> str:
        return f"{path}.scaler.joblib"

    def save(self, path: str):
        import joblib
        self.model.save(path)                 # keras SavedModel/.keras
        joblib.dump({"mu": self._mu, "sigma": self._sigma}, self._scaler_path(path))

    @classmethod
    def load(cls, path: str):
        import joblib
        from tensorflow.keras.models import load_model
        o = cls()
        o.model = load_model(path)
        o.horizon = o.model.output_shape[-1]
        stats = joblib.load(cls._scaler_path(path))
        o._mu, o._sigma = stats["mu"], stats["sigma"]
        return o