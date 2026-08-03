"""Serving-side forecaster: loads the approved model + preprocessor and
produces a horizon-length forecast, plus simple business recommendations.

Model-agnostic: it reads which model is approved from summary.json (Phase 1) or
the MLflow registry (Phase 2), so swapping models needs no API change.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from forecasting.core.registry import get_model
from forecasting.implementations.models import forecasters  # noqa: F401 (register)
from forecasting.implementations.preprocessors.generic import (
    CalendarFeatureEngineer,
    GenericPreprocessor,
    SlidingWindowBuilder,
)


class Forecaster:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.dir = artifacts_dir
        with open(os.path.join(artifacts_dir, "summary.json")) as f:
            self.summary = json.load(f)
        self.best = self.summary["best_model"]
        self.horizon = self.summary["horizon"]
        self.lookback = self.summary["lookback"]

        self.pre = GenericPreprocessor.load(os.path.join(artifacts_dir, "preprocessor.joblib"))
        self.schema = self.pre.schema
        ext = ".keras" if self.best == "lstm" else ".joblib"
        self.model = get_model(self.best).load(
            os.path.join(artifacts_dir, f"model_{self.best}{ext}")
        )
        self.fe = CalendarFeatureEngineer()
        self.wb = SlidingWindowBuilder(lookback=self.lookback, horizon=self.horizon)

    def forecast(self, history: pd.DataFrame) -> dict:
        """history: recent rows for ONE series (>= lookback+features). Returns
        the next `horizon` predicted values + dates + business insight."""
        df_p = self.pre.transform(history)
        df_f = self.fe.engineer(df_p, self.schema)
        X_tab, X_seq, _, _ = self.wb.build(df_f, self.schema)
        if len(X_tab) == 0:
            raise ValueError("Not enough history to build a forecast window.")
        X = (X_seq if self.best == "lstm" else X_tab)[-1:]     # most recent window
        preds = self.model.predict(X)[0]
        preds = np.clip(preds, 0, None)

        last_date = pd.to_datetime(df_f[self.schema.date_column]).max()
        future_dates = pd.date_range(last_date, periods=self.horizon + 1, freq="D")[1:]

        recent_avg = float(df_f[self.schema.target_column].tail(self.lookback).mean())
        forecast_avg = float(np.mean(preds))
        delta = (forecast_avg - recent_avg) / max(recent_avg, 1e-6)

        return {
            "model": self.best,
            "horizon": self.horizon,
            "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
            "forecast": [round(float(p), 1) for p in preds],
            "recent_average": round(recent_avg, 1),
            "forecast_average": round(forecast_avg, 1),
            "trend_pct": round(delta * 100, 1),
            "insight": self._insight(delta),
        }

    @staticmethod
    def _insight(delta: float) -> str:
        pct = abs(round(delta * 100))
        if delta > 0.05:
            return f"Demand trending UP ~{pct}% — consider increasing inventory & staffing."
        if delta < -0.05:
            return f"Demand trending DOWN ~{pct}% — consider reducing stock to avoid waste."
        return "Demand roughly stable — maintain current inventory levels."