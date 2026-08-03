"""Generic, schema-driven preprocessing + feature engineering + windowing.

No hardcoded column names — everything is driven by the DatasetSchema, so the
same code works for Rossmann or any other business time-series dataset.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from forecasting.interfaces.contracts import (
    DatasetSchema,
    IFeatureEngineer,
    IPreprocessor,
    IWindowBuilder,
)


class GenericPreprocessor(IPreprocessor):
    """Parse dates, impute, encode categoricals, scale numerics."""

    def __init__(self):
        self.scaler: StandardScaler | None = None
        self.encoder: OrdinalEncoder | None = None
        self.schema: DatasetSchema | None = None
        self._num_cols: list[str] = []
        self._cat_cols: list[str] = []

    def fit(self, df: pd.DataFrame, schema: DatasetSchema) -> "GenericPreprocessor":
        self.schema = schema
        df = self._parse_dates(df.copy(), schema)
        self._num_cols = [c for c in schema.numerical_features if c in df.columns]
        self._cat_cols = [c for c in schema.categorical_features if c in df.columns]

        if self._num_cols:
            self.scaler = StandardScaler().fit(self._impute(df[self._num_cols]))
        if self._cat_cols:
            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            ).fit(df[self._cat_cols].astype(str).fillna("NA"))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._parse_dates(df.copy(), self.schema)
        if self._num_cols:
            df[self._num_cols] = self.scaler.transform(self._impute(df[self._num_cols]))
        if self._cat_cols:
            df[self._cat_cols] = self.encoder.transform(
                df[self._cat_cols].astype(str).fillna("NA")
            )
        # sort by (group, date) so windows are chronological
        sort_cols = [c for c in [self.schema.group_column, self.schema.date_column] if c]
        return df.sort_values(sort_cols).reset_index(drop=True)

    def _parse_dates(self, df, schema):
        df[schema.date_column] = pd.to_datetime(df[schema.date_column], errors="coerce")
        return df.dropna(subset=[schema.date_column])

    @staticmethod
    def _impute(x: pd.DataFrame) -> pd.DataFrame:
        return x.fillna(x.median(numeric_only=True))

    def save(self, path: str):
        joblib.dump(
            {"scaler": self.scaler, "encoder": self.encoder, "schema": self.schema,
             "num": self._num_cols, "cat": self._cat_cols}, path
        )

    @classmethod
    def load(cls, path: str) -> "GenericPreprocessor":
        d = joblib.load(path)
        o = cls()
        o.scaler, o.encoder, o.schema = d["scaler"], d["encoder"], d["schema"]
        o._num_cols, o._cat_cols = d["num"], d["cat"]
        return o


class CalendarFeatureEngineer(IFeatureEngineer):
    """Derive calendar + lag + rolling features from the date/target."""

    def __init__(self, lags=(1, 7, 14), rolls=(7, 14)):
        self.lags = lags
        self.rolls = rolls

    def engineer(self, df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
        df = df.copy()
        d = pd.to_datetime(df[schema.date_column])
        df["dow"] = d.dt.dayofweek
        df["day"] = d.dt.day
        df["month"] = d.dt.month
        df["year"] = d.dt.year
        df["weekofyear"] = d.dt.isocalendar().week.astype(int)
        df["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)

        tgt = schema.target_column
        grp = schema.group_column
        g = df.groupby(grp)[tgt] if grp else df[tgt]

        for lag in self.lags:
            df[f"{tgt}_lag{lag}"] = g.shift(lag) if grp else df[tgt].shift(lag)
        for r in self.rolls:
            if grp:
                df[f"{tgt}_roll{r}"] = (
                    df.groupby(grp)[tgt].shift(1).rolling(r).mean().reset_index(0, drop=True)
                )
            else:
                df[f"{tgt}_roll{r}"] = df[tgt].shift(1).rolling(r).mean()

        return df.dropna().reset_index(drop=True)


class SlidingWindowBuilder(IWindowBuilder):
    """Build (X, y) for a multi-step horizon.

    Returns:
      X_tab : 2D array for tabular models (flattened window + current features)
      X_seq : 3D array (samples, lookback, features) for sequence models
      y     : 2D array (samples, horizon)
    """

    def __init__(self, lookback: int = 14, horizon: int = 7):
        self.lookback = lookback
        self.horizon = horizon

    def build(self, df: pd.DataFrame, schema: DatasetSchema):
        tgt = schema.target_column
        feat_cols = [
            c for c in df.columns
            if c not in {schema.date_column, tgt, schema.group_column}
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        series_target = df[tgt].values.astype(float)
        feats = df[feat_cols].values.astype(float)

        X_seq, X_tab, y = [], [], []
        n = len(df)
        for i in range(self.lookback, n - self.horizon + 1):
            past = series_target[i - self.lookback:i]                 # lookback window
            cur_feats = feats[i]                                       # features at t
            target = series_target[i:i + self.horizon]                # next horizon
            X_seq.append(np.column_stack([
                series_target[i - self.lookback:i].reshape(-1, 1),
                feats[i - self.lookback:i],
            ]))
            X_tab.append(np.concatenate([past, cur_feats]))
            y.append(target)

        return (
            np.array(X_tab, dtype="float32"),
            np.array(X_seq, dtype="float32"),
            np.array(y, dtype="float32"),
            feat_cols,
        )