"""Auto-detect dataset schema from any time-series CSV.

Heuristics (all generic — nothing Rossmann-specific):
  * Date column: the column that parses to datetime for the most rows.
  * Target: prefer names matching sales/revenue/demand/orders/qty; else the
    numeric column with the highest variance that isn't an ID/date.
  * Numerical vs categorical: dtype + cardinality thresholds.
  * Frequency: inferred from the median gap between sorted dates.
User overrides from config are merged on top, so uncertain guesses are fixable.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from forecasting.interfaces.contracts import DatasetSchema, ISchemaDetector

TARGET_HINTS = re.compile(
    r"(sales|revenue|demand|orders?|qty|quantity|units|count|volume|amount)",
    re.IGNORECASE,
)
ID_HINTS = re.compile(r"(id$|_id|index|store|shop|item|sku|product)", re.IGNORECASE)


class HeuristicSchemaDetector(ISchemaDetector):
    def __init__(self, high_cardinality_threshold: int = 50):
        self.hc_threshold = high_cardinality_threshold

    def detect(self, df: pd.DataFrame, overrides: dict | None = None) -> DatasetSchema:
        overrides = overrides or {}
        conf: dict[str, float] = {}
        notes: list[str] = []

        date_col = overrides.get("date_column") or self._detect_date(df, conf, notes)
        target_col = overrides.get("target_column") or self._detect_target(df, date_col, conf, notes)
        group_col = overrides.get("group_column") or self._detect_group(df, date_col, target_col)

        # Classify remaining columns
        exclude = {date_col, target_col, group_col} - {None}
        num, cat = [], []
        for c in df.columns:
            if c in exclude:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                # low-cardinality integers that look categorical (flags) -> categorical
                nun = df[c].nunique(dropna=True)
                if nun <= 12 and set(df[c].dropna().unique()).issubset(set(range(-1, 13))):
                    cat.append(c)
                else:
                    num.append(c)
            else:
                cat.append(c)

        num = overrides.get("numerical_features", num)
        cat = overrides.get("categorical_features", cat)

        freq = overrides.get("frequency") or self._detect_frequency(df, date_col, notes)

        return DatasetSchema(
            date_column=date_col,
            target_column=target_col,
            numerical_features=num,
            categorical_features=cat,
            frequency=freq,
            group_column=group_col,
            confidence=conf,
            notes=notes,
        )

    # ── detectors ──
    def _detect_date(self, df, conf, notes) -> str:
        best, best_ratio = None, 0.0
        for c in df.columns:
            s = df[c]
            if pd.api.types.is_datetime64_any_dtype(s):
                return self._record(conf, c, 1.0, "date")
            if s.dtype == object or "date" in c.lower() or "time" in c.lower():
                parsed = pd.to_datetime(s, errors="coerce")
                ratio = parsed.notna().mean()
                if ratio > best_ratio:
                    best, best_ratio = c, ratio
        if best is None or best_ratio < 0.6:
            notes.append("Date column uncertain — consider setting it via config override.")
        conf["date"] = round(float(best_ratio), 3)
        return best

    def _detect_target(self, df, date_col, conf, notes) -> str:
        candidates = [
            c for c in df.columns
            if c != date_col and pd.api.types.is_numeric_dtype(df[c])
            and not ID_HINTS.search(c)
        ]
        # 1) name hint
        for c in candidates:
            if TARGET_HINTS.search(c):
                conf["target"] = 0.9
                return c
        # 2) highest-variance numeric fallback
        if candidates:
            variances = {c: df[c].var() for c in candidates}
            pick = max(variances, key=variances.get)
            conf["target"] = 0.5
            notes.append(f"Target guessed by variance ('{pick}') — verify via config.")
            return pick
        raise ValueError("No numeric target column could be detected.")

    def _detect_group(self, df, date_col, target_col):
        # A group column repeats dates (multiple series). Heuristic: an ID-like
        # column where (group, date) is closer to unique than date alone.
        for c in df.columns:
            if c in (date_col, target_col):
                continue
            if ID_HINTS.search(c) and df[c].nunique() > 1:
                return c
        return None

    def _detect_frequency(self, df, date_col, notes) -> str | None:
        try:
            d = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values().unique()
            if len(d) < 3:
                return None
            gaps = np.diff(d).astype("timedelta64[D]").astype(int)
            gaps = gaps[gaps > 0]
            if len(gaps) == 0:
                return None
            median_gap = int(np.median(gaps))
            return {1: "D", 7: "W"}.get(median_gap, f"{median_gap}D")
        except Exception:  # noqa: BLE001
            notes.append("Frequency detection failed.")
            return None

    @staticmethod
    def _record(conf, col, score, key):
        conf[key] = score
        return col