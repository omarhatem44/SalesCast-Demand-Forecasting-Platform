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
    """Sequence-to-vector LSTM: lookback window -> horizon-length forecast."""

    def __init__(self, horizon: int = 7, units: int = 64, epochs: int = 15,
                 batch_size: int = 64):
        self.horizon = horizon
        self.units = units
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None

    def _build(self, n_timesteps, n_features):
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential
        m = Sequential([
            LSTM(self.units, input_shape=(n_timesteps, n_features)),
            Dropout(0.2),
            Dense(self.units // 2, activation="relu"),
            Dense(self.horizon),
        ])
        m.compile(optimizer="adam", loss="mse", metrics=["mae"])
        return m

    def fit(self, X, y):
        y = np.asarray(y)
        self.horizon = y.shape[1]
        self.model = self._build(X.shape[1], X.shape[2])
        self.model.fit(
            X, y, epochs=self.epochs, batch_size=self.batch_size,
            validation_split=0.1, verbose=0,
        )
        return self

    def predict(self, X):
        return self.model.predict(X, verbose=0)

    def save(self, path: str):
        self.model.save(path)                 # keras SavedModel/.keras

    @classmethod
    def load(cls, path: str):
        from tensorflow.keras.models import load_model
        o = cls()
        o.model = load_model(path)
        o.horizon = o.model.output_shape[-1]
        return o