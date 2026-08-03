"""Standardized evaluation (MAE, RMSE, RMSPE) and best-model selection."""
from __future__ import annotations

import numpy as np

from forecasting.interfaces.contracts import (
    EvalResult,
    IModel,
    IModelEvaluator,
    IModelSelector,
)


def _rmspe(y_true, y_pred, eps=1e-6):
    """Root Mean Square Percentage Error — the Rossmann metric.
    Days/series with true value 0 are ignored (standard for RMSPE)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    yt, yp = y_true[mask], y_pred[mask]
    return float(np.sqrt(np.mean(((yt - yp) / (yt + eps)) ** 2)))


class StandardEvaluator(IModelEvaluator):
    def evaluate(self, model: IModel, X, y_true) -> EvalResult:
        y_pred = model.predict(X)
        yt = np.asarray(y_true, dtype=float).ravel()
        yp = np.asarray(y_pred, dtype=float).ravel()
        mae = float(np.mean(np.abs(yt - yp)))
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        rmspe = _rmspe(y_true, y_pred)
        return EvalResult(model_name=model.name, mae=mae, rmse=rmse, rmspe=rmspe)


class BestByMetricSelector(IModelSelector):
    """Select the model minimizing a chosen metric (default RMSPE)."""

    def __init__(self, metric: str = "rmspe"):
        self.metric = metric

    def select(self, results: list[EvalResult]) -> str:
        valid = [r for r in results if not np.isnan(getattr(r, self.metric))]
        if not valid:
            # fall back to RMSE if the chosen metric is all-NaN
            valid, self.metric = results, "rmse"
        best = min(valid, key=lambda r: getattr(r, self.metric))
        return best.model_name