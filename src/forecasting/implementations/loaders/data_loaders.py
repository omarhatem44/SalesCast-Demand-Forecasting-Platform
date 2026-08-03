"""Generic data profiler + CSV loader."""
from __future__ import annotations

import pandas as pd

from forecasting.interfaces.contracts import (
    DataProfile,
    IDataLoader,
    IDataProfiler,
)


class CsvLoader(IDataLoader):
    def __init__(self, path: str, **read_kwargs):
        self.path = path
        self.read_kwargs = read_kwargs

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path, **self.read_kwargs)


class DataFrameLoader(IDataLoader):
    """Wrap an in-memory frame (used for uploads in Phase 3)."""
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def load(self) -> pd.DataFrame:
        return self._df.copy()


class BasicProfiler(IDataProfiler):
    def __init__(self, high_cardinality_threshold: int = 50):
        self.hc = high_cardinality_threshold

    def profile(self, df: pd.DataFrame) -> DataProfile:
        warnings: list[str] = []
        missing = df.isna().sum().to_dict()
        constant = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        high_card = [
            c for c in df.columns
            if df[c].dtype == object and df[c].nunique() > self.hc
        ]
        dups = int(df.duplicated().sum())

        if constant:
            warnings.append(f"Constant columns (no signal): {constant}")
        if dups:
            warnings.append(f"{dups} duplicate rows.")
        heavy_missing = [c for c, m in missing.items() if m > 0.4 * len(df)]
        if heavy_missing:
            warnings.append(f"Columns >40% missing: {heavy_missing}")

        stats = {
            c: {
                "dtype": str(df[c].dtype),
                "nunique": int(df[c].nunique(dropna=True)),
                "missing": int(missing[c]),
            }
            for c in df.columns
        }

        return DataProfile(
            n_rows=len(df),
            n_columns=df.shape[1],
            missing={k: int(v) for k, v in missing.items()},
            constant_columns=constant,
            high_cardinality=high_card,
            duplicate_rows=dups,
            statistics=stats,
            warnings=warnings,
        )