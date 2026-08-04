"""Drift monitoring (Phase 2).

Implements IMonitor. Primary path uses Evidently AI; if Evidently isn't
installed or its API differs across versions, falls back to a lightweight
built-in PSI (Population Stability Index) + KS-style check so monitoring always
works. This mirrors the defensive pattern used elsewhere in the platform.

The monitor compares a reference dataset (training distribution) against a
current dataset (recent production data) and reports per-column drift plus an
overall dataset-drift flag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.interfaces.contracts import DatasetSchema, IMonitor


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two numeric distributions.
    PSI < 0.1 = no significant shift, 0.1–0.25 = moderate, > 0.25 = major."""
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 2:
        return 0.0
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


class SimpleDriftMonitor(IMonitor):
    """Built-in PSI-based drift monitor — no external dependencies."""

    def __init__(self, psi_threshold: float = 0.25, drift_share: float = 0.4):
        self.psi_threshold = psi_threshold
        self.drift_share = drift_share

    def check_drift(self, reference: pd.DataFrame, current: pd.DataFrame) -> dict:
        cols = [
            c for c in reference.columns
            if c in current.columns and pd.api.types.is_numeric_dtype(reference[c])
        ]
        per_col = {}
        drifted = 0
        for c in cols:
            score = _psi(reference[c].to_numpy(float), current[c].to_numpy(float))
            is_drift = score > self.psi_threshold
            per_col[c] = {"psi": round(score, 4), "drifted": bool(is_drift)}
            drifted += int(is_drift)

        n = max(len(cols), 1)
        share = drifted / n
        return {
            "method": "psi",
            "n_columns": n,
            "n_drifted": drifted,
            "drift_share": round(share, 3),
            "dataset_drift": bool(share >= self.drift_share),
            "columns": per_col,
        }


class EvidentlyDriftMonitor(IMonitor):
    """Evidently-backed monitor with a fallback to SimpleDriftMonitor.

    Tries the Evidently API (both v0.7+ and older layouts). If Evidently isn't
    available or errors, transparently falls back to the built-in PSI monitor,
    so the /model-health endpoint never breaks.
    """

    def __init__(self, schema: DatasetSchema | None = None):
        self.schema = schema
        self._fallback = SimpleDriftMonitor()

    def check_drift(self, reference: pd.DataFrame, current: pd.DataFrame) -> dict:
        try:
            return self._evidently(reference, current)
        except Exception as e:  # noqa: BLE001
            result = self._fallback.check_drift(reference, current)
            result["method"] = "psi (evidently unavailable)"
            result["fallback_reason"] = str(e)[:200]
            return result

    def _evidently(self, reference: pd.DataFrame, current: pd.DataFrame) -> dict:
        # New API (v0.7+): evidently.Report + presets
        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset

            report = Report([DataDriftPreset()])
            snapshot = report.run(current, reference)
            d = snapshot.dict() if hasattr(snapshot, "dict") else {}
            return self._parse_new(d)
        except ImportError:
            pass

        # Older API (<=0.6): evidently.report.Report + metric_preset
        from evidently.metric_preset import DataDriftPreset  # type: ignore
        from evidently.report import Report  # type: ignore

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        d = report.as_dict()
        drift = d["metrics"][0]["result"]
        return {
            "method": "evidently",
            "n_columns": drift.get("number_of_columns"),
            "n_drifted": drift.get("number_of_drifted_columns"),
            "drift_share": round(drift.get("share_of_drifted_columns", 0), 3),
            "dataset_drift": bool(drift.get("dataset_drift", False)),
            "columns": {},
        }

    @staticmethod
    def _parse_new(d: dict) -> dict:
        # The v0.7 snapshot dict structure varies; extract defensively.
        metrics = d.get("metrics", []) if isinstance(d, dict) else []
        share, n_drift, n_cols = 0.0, None, None
        for m in metrics:
            res = m.get("result", m) if isinstance(m, dict) else {}
            if "share_of_drifted_columns" in res:
                share = res["share_of_drifted_columns"]
                n_drift = res.get("number_of_drifted_columns")
                n_cols = res.get("number_of_columns")
        return {
            "method": "evidently",
            "n_columns": n_cols,
            "n_drifted": n_drift,
            "drift_share": round(share, 3),
            "dataset_drift": bool(share >= 0.4),
            "columns": {},
        }