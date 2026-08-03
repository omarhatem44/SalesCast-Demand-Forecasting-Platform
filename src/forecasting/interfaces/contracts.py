"""Abstract interfaces for the forecasting platform.

Every major stage is defined as an ABC so implementations are interchangeable
(Dependency Inversion). New models, loaders, or preprocessors can be added by
implementing the relevant interface and registering them — no existing code
needs to change (Open/Closed).

These contracts are dataset-agnostic on purpose: nothing here mentions Rossmann.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ─────────────────────────────────────────────────────────────
# Shared data contracts passed between stages
# ─────────────────────────────────────────────────────────────
@dataclass
class DatasetSchema:
    """The detected (or user-overridden) structure of a dataset."""
    date_column: str
    target_column: str
    numerical_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    frequency: str | None = None            # 'D', 'W', 'M', ...
    group_column: str | None = None         # e.g. store id, for multi-series
    confidence: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class DataProfile:
    """Output of profiling: statistics + quality flags."""
    n_rows: int
    n_columns: int
    missing: dict[str, int] = field(default_factory=dict)
    constant_columns: list[str] = field(default_factory=list)
    high_cardinality: list[str] = field(default_factory=list)
    duplicate_rows: int = 0
    statistics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Standardized metrics for one model, so models are comparable."""
    model_name: str
    mae: float
    rmse: float
    rmspe: float
    extra: dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────
# Interfaces
# ─────────────────────────────────────────────────────────────
class IDataLoader(ABC):
    """Load raw data from some source into a DataFrame."""

    @abstractmethod
    def load(self) -> pd.DataFrame: ...


class IDataProfiler(ABC):
    """Inspect a DataFrame and report statistics + quality issues."""

    @abstractmethod
    def profile(self, df: pd.DataFrame) -> DataProfile: ...


class ISchemaDetector(ABC):
    """Infer the dataset schema (date/target/feature roles).

    Implementations may auto-detect; a config override is merged on top so the
    user can correct uncertain guesses.
    """

    @abstractmethod
    def detect(self, df: pd.DataFrame, overrides: dict | None = None) -> DatasetSchema: ...


class IPreprocessor(ABC):
    """Clean + transform raw data into a modeling-ready frame.

    Parse dates, handle missing values, encode categoricals, scale numerics —
    all driven by the DatasetSchema, never by hardcoded column names.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame, schema: DatasetSchema) -> "IPreprocessor": ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def fit_transform(self, df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
        return self.fit(df, schema).transform(df)


class IFeatureEngineer(ABC):
    """Derive features (calendar, lags, rolling stats) from the clean frame."""

    @abstractmethod
    def engineer(self, df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame: ...


class IWindowBuilder(ABC):
    """Turn a feature frame into supervised (X, y) windows for a horizon.

    Used by sequence models (LSTM) and, in flattened form, by tabular models.
    """

    @abstractmethod
    def build(self, df: pd.DataFrame, schema: DatasetSchema): ...


class IModel(ABC):
    """A forecasting model. All models expose the same fit/predict surface so
    the selector can compare them and the API can load any of them generically.
    """

    name: str

    @abstractmethod
    def fit(self, X, y) -> "IModel": ...

    @abstractmethod
    def predict(self, X): ...

    @abstractmethod
    def save(self, path: str) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "IModel": ...


class IModelEvaluator(ABC):
    """Compute standardized metrics (MAE, RMSE, RMSPE) for a model."""

    @abstractmethod
    def evaluate(self, model: IModel, X, y_true) -> EvalResult: ...


class IModelSelector(ABC):
    """Pick the best model from a set of EvalResults by a chosen metric."""

    @abstractmethod
    def select(self, results: list[EvalResult]) -> str: ...


class IForecaster(ABC):
    """Produce a multi-step forecast from the currently approved model."""

    @abstractmethod
    def forecast(self, history: pd.DataFrame, horizon: int) -> pd.DataFrame: ...


class IMonitor(ABC):
    """(Phase 2) Detect data / prediction drift."""

    @abstractmethod
    def check_drift(self, reference: pd.DataFrame, current: pd.DataFrame) -> dict: ...


class IRetrainer(ABC):
    """(Phase 2) Decide whether to retrain and trigger it."""

    @abstractmethod
    def should_retrain(self, drift_report: dict, metrics: EvalResult) -> bool: ...