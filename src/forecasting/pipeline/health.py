"""Model-health service (Phase 2).

Aggregates what the dashboard's 'Model Health' panel needs:
  * approved model + version
  * last retrain timestamp
  * data drift + prediction drift status
  * latest evaluation metrics

Reference data = a saved sample of the training distribution.
Current data   = recent history sent to the forecaster (or a supplied window).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd

from forecasting.implementations.monitors.drift import EvidentlyDriftMonitor
from forecasting.implementations.preprocessors.generic import GenericPreprocessor


class ModelHealthService:
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.dir = artifacts_dir
        with open(os.path.join(artifacts_dir, "summary.json")) as f:
            self.summary = json.load(f)
        self.schema = None
        pre_path = os.path.join(artifacts_dir, "preprocessor.joblib")
        if os.path.exists(pre_path):
            self.schema = GenericPreprocessor.load(pre_path).schema
        self.monitor = EvidentlyDriftMonitor(self.schema)

        # reference sample saved at train time (if present)
        self._ref_path = os.path.join(artifacts_dir, "reference_sample.csv")

    def _last_retrain(self) -> str:
        model_files = [f for f in os.listdir(self.dir) if f.startswith("model_")]
        if not model_files:
            return "unknown"
        newest = max(
            os.path.getmtime(os.path.join(self.dir, f)) for f in model_files
        )
        return datetime.fromtimestamp(newest, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def _model_version(self) -> str:
        # Phase 2: derive from artifact mtime; Phase 2b: read from MLflow registry.
        return self.summary.get("model_version", "v1")

    def health(self, current: pd.DataFrame | None = None) -> dict:
        base = {
            "approved_model": self.summary.get("best_model"),
            "model_version": self._model_version(),
            "last_retrain": self._last_retrain(),
            "metrics": self.summary.get("metrics", {}),
        }

        # Drift, if we have both a reference sample and current data.
        if current is not None and os.path.exists(self._ref_path):
            try:
                reference = pd.read_csv(self._ref_path)
                target = self.schema.target_column if self.schema else None
                data_drift = self.monitor.check_drift(reference, current)
                base["data_drift"] = {
                    "detected": data_drift["dataset_drift"],
                    "share": data_drift["drift_share"],
                    "method": data_drift["method"],
                }
                # prediction drift = drift specifically on the target column
                if target and target in reference and target in current:
                    pred = self.monitor.check_drift(
                        reference[[target]], current[[target]]
                    )
                    base["prediction_drift"] = {
                        "detected": pred["dataset_drift"],
                        "share": pred["drift_share"],
                    }
            except Exception as e:  # noqa: BLE001
                base["drift_error"] = str(e)[:200]
        else:
            base["data_drift"] = {"detected": None, "note": "no reference/current data"}

        base["status"] = self._status(base)
        return base

    @staticmethod
    def _status(base: dict) -> str:
        dd = base.get("data_drift", {})
        if dd.get("detected") is True:
            return "DRIFT DETECTED — retraining recommended"
        if dd.get("detected") is False:
            return "HEALTHY — no significant drift"
        return "MONITORING — awaiting production data"