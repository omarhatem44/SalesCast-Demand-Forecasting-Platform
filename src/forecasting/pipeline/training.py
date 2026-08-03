"""End-to-end training pipeline — wires the interfaces together.

Depends only on the ABCs, so any implementation can be swapped via config.
Flow: load → profile → detect schema → preprocess → engineer → window →
train each configured model → evaluate → select best → persist artifacts.
"""
from __future__ import annotations

import json
import os

import numpy as np

from forecasting.core.registry import get_model
from forecasting.implementations.evaluators.standard import (
    BestByMetricSelector,
    StandardEvaluator,
)
from forecasting.implementations.loaders.data_loaders import BasicProfiler, CsvLoader
from forecasting.implementations.models import forecasters  # noqa: F401  (registers models)
from forecasting.implementations.preprocessors.generic import (
    CalendarFeatureEngineer,
    GenericPreprocessor,
    SlidingWindowBuilder,
)
from forecasting.implementations.profilers.schema_detector import HeuristicSchemaDetector


class TrainingPipeline:
    def __init__(self, config: dict):
        self.cfg = config
        self.artifacts = config.get("artifacts_dir", "artifacts")
        os.makedirs(self.artifacts, exist_ok=True)

    def run(self) -> dict:
        cfg = self.cfg
        horizon = cfg.get("horizon", 7)
        lookback = cfg.get("lookback", 14)

        # 1. Load
        df = CsvLoader(cfg["data_path"], nrows=cfg.get("nrows")).load()

        # 2. Profile
        profile = BasicProfiler().profile(df)

        # 3. Detect schema (+ user overrides)
        schema = HeuristicSchemaDetector().detect(df, overrides=cfg.get("schema_overrides"))
        print(f"[schema] date={schema.date_column} target={schema.target_column} "
              f"group={schema.group_column} freq={schema.frequency}")
        print(f"[schema] {len(schema.numerical_features)} numeric, "
              f"{len(schema.categorical_features)} categorical")

        # Optional: focus on a single series for a manageable MVP
        if cfg.get("single_group") and schema.group_column:
            gval = cfg["single_group"]
            df = df[df[schema.group_column] == gval].copy()
            print(f"[data] restricted to {schema.group_column}={gval}: {len(df)} rows")

        # 4. Preprocess
        pre = GenericPreprocessor()
        df_p = pre.fit_transform(df, schema)

        # 5. Feature engineering
        fe = CalendarFeatureEngineer()
        df_f = fe.engineer(df_p, schema)

        # 6. Window
        wb = SlidingWindowBuilder(lookback=lookback, horizon=horizon)
        X_tab, X_seq, y, feat_cols = wb.build(df_f, schema)
        if len(y) < 30:
            raise ValueError(f"Too few windows ({len(y)}) — need more history.")

        # chronological split
        split = int(len(y) * 0.8)
        results, trained = [], {}
        evaluator = StandardEvaluator()

        for name in cfg.get("models", ["xgboost", "lstm"]):
            ModelCls = get_model(name)
            model = ModelCls(horizon=horizon)
            X = X_seq if name == "lstm" else X_tab
            model.fit(X[:split], y[:split])
            res = evaluator.evaluate(model, X[split:], y[split:])
            results.append(res)
            trained[name] = (model, X)
            print(f"[eval] {name:8s} MAE={res.mae:.2f} RMSE={res.rmse:.2f} "
                  f"RMSPE={res.rmspe:.4f}")

        # 7. Select best
        best_name = BestByMetricSelector(cfg.get("selection_metric", "rmspe")).select(results)
        print(f"[select] best model: {best_name}")

        # 8. Persist
        pre.save(os.path.join(self.artifacts, "preprocessor.joblib"))
        best_model, _ = trained[best_name]
        ext = ".keras" if best_name == "lstm" else ".joblib"
        best_model.save(os.path.join(self.artifacts, f"model_{best_name}{ext}"))

        summary = {
            "best_model": best_name,
            "horizon": horizon,
            "lookback": lookback,
            "metrics": {r.model_name: {"mae": r.mae, "rmse": r.rmse, "rmspe": r.rmspe}
                        for r in results},
            "schema": {
                "date_column": schema.date_column,
                "target_column": schema.target_column,
                "group_column": schema.group_column,
                "frequency": schema.frequency,
            },
            "profile_warnings": profile.warnings,
        }
        with open(os.path.join(self.artifacts, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[done] artifacts -> {self.artifacts}")
        return summary