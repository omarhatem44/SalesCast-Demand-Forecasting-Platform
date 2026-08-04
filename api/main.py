"""SalesCast Forecasting API (FastAPI).

Endpoints:
  GET  /health              -> liveness
  GET  /model-info          -> which model is approved, metrics, schema
  POST /forecast            -> 7-day forecast + business insight for a series
  GET  /                    -> serves the dashboard

The API always loads whatever model is currently approved (Phase 1: summary.json;
Phase 2: MLflow registry) — no model name is hardcoded here.
"""
from __future__ import annotations

import io
import os
import sys

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, "src")
from forecasting.pipeline.forecaster import Forecaster  # noqa: E402

ARTIFACTS = os.environ.get("ARTIFACTS_DIR", "artifacts")

app = FastAPI(title="SalesCast Forecasting Platform", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_forecaster: Forecaster | None = None


def get_forecaster() -> Forecaster:
    global _forecaster
    if _forecaster is None:
        _forecaster = Forecaster(ARTIFACTS)
    return _forecaster


class ForecastRequest(BaseModel):
    # A list of recent history records for ONE series (store/product/etc.)
    history: list[dict]


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/model-info")
def model_info():
    f = get_forecaster()
    return {
        "approved_model": f.best,
        "horizon": f.horizon,
        "lookback": f.lookback,
        "metrics": f.summary.get("metrics", {}),
        "schema": f.summary.get("schema", {}),
    }


@app.post("/forecast")
def forecast(req: ForecastRequest):
    try:
        hist = pd.DataFrame(req.history)
        return get_forecaster().forecast(hist)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model-health")
def model_health():
    """Phase 2: drift status + model metadata for the dashboard health panel."""
    from forecasting.pipeline.health import ModelHealthService
    svc = ModelHealthService(ARTIFACTS)
    # Use the reference sample itself as 'current' when no live data is posted,
    # so the panel shows a real (no-drift) baseline instead of empty state.
    return svc.health(current=None)


@app.post("/model-health")
def model_health_with_data(req: ForecastRequest):
    """Drift check against posted recent production data."""
    import pandas as pd
    from forecasting.pipeline.health import ModelHealthService
    svc = ModelHealthService(ARTIFACTS)
    cur = pd.DataFrame(req.history)
    # engineer current data the same way so columns line up with the reference
    from forecasting.implementations.preprocessors.generic import (
        CalendarFeatureEngineer, GenericPreprocessor,
    )
    pre = GenericPreprocessor.load(os.path.join(ARTIFACTS, "preprocessor.joblib"))
    cur_f = CalendarFeatureEngineer().engineer(pre.transform(cur), pre.schema)
    return svc.health(current=cur_f)


@app.get("/")
def home():
    idx = os.path.join("dashboard", "index.html")
    if os.path.exists(idx):
        return FileResponse(idx)
    return {"message": "SalesCast API running. Dashboard not found."}