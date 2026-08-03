<div align="center">

# 📈 SalesCast — Demand Forecasting Platform

**A reusable, model-agnostic forecasting platform** — not a single-model demo.
Upload any business time-series, and the platform detects its schema, engineers features,
trains and compares multiple models, auto-selects the best, and serves 7-day forecasts
with business recommendations.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Baseline-EB5E28?style=for-the-badge)](https://xgboost.ai)
[![TensorFlow](https://img.shields.io/badge/LSTM-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-Registry-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)

</div>

---

## 🎯 What makes this a *platform*, not a project

Most forecasting projects hardcode their dataset. SalesCast is built around **SOLID interfaces**
so every stage is replaceable and the pipeline works on **any** business time-series with minimal config:

```
CSV → Profile → Schema Detection → Preprocess → Feature Engineering →
      [ XGBoost | LSTM | (future: ARIMA, Prophet, TFT…) ] →
      Evaluate (MAE/RMSE/RMSPE) → Auto-select best → MLflow → Forecast API → Dashboard
```

**Nothing is Rossmann-specific.** Schema detection finds the date/target/feature roles
automatically (with config override for uncertain cases), so the same code forecasts retail
sales, energy demand, restaurant orders, or supply-chain volume.

---

## 🏗️ Architecture (SOLID / Clean)

Every major component is an interface (`abc.ABC`) with interchangeable implementations,
wired via a config-driven registry. Add a new model or preprocessor **without modifying existing code** (Open/Closed).

| Interface | Purpose | Phase 1 implementation |
|---|---|---|
| `IDataLoader` | load raw data | `CsvLoader`, `DataFrameLoader` |
| `IDataProfiler` | stats + quality flags | `BasicProfiler` |
| `ISchemaDetector` | auto-detect date/target/features | `HeuristicSchemaDetector` |
| `IPreprocessor` | clean/encode/scale (schema-driven) | `GenericPreprocessor` |
| `IFeatureEngineer` | calendar/lag/rolling features | `CalendarFeatureEngineer` |
| `IWindowBuilder` | sliding windows for sequences | `SlidingWindowBuilder` |
| `IModel` | fit/predict/save/load | `XGBoostForecaster`, `LSTMForecaster` |
| `IModelEvaluator` | MAE/RMSE/RMSPE | `StandardEvaluator` |
| `IModelSelector` | pick best by metric | `BestByMetricSelector` |
| `IForecaster` | serve N-day forecast + insight | `Forecaster` |
| `IMonitor` *(Phase 2)* | drift detection | Evidently AI |
| `IRetrainer` *(Phase 2)* | retrain trigger | Airflow DAG |

New models self-register:
```python
@register_model("prophet")
class ProphetForecaster(IModel):
    ...
# instantly available to the pipeline & selector — zero other changes
```

---

## 🧠 Modeling — honest, not dogmatic

The platform trains **both** XGBoost and LSTM, evaluates on **MAE / RMSE / RMSPE**, and
**auto-selects whichever actually wins** — it does not force deep learning. On some datasets
gradient boosting beats the LSTM, and the system reports that truthfully. The prediction API
always loads the **currently approved model** from the registry, so swapping the winner needs
no API change.

---

## 📊 Dashboard

A retail-analytics interface (not just a chart): store selector, historical sales, 7-day
forecast with confidence band, inventory recommendations, and a model-health panel.

> Inventory Recommendation (derived from forecast trend):
> Milk → increase 18% · Coffee → increase 7% · Bread → reduce 5%

<!-- Add a screenshot: save as assets/dashboard.png and uncomment -->
<!-- <img src="assets/dashboard.png" width="900"/> -->

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# drop the real Rossmann train.csv in data/  (or use the included synthetic sample)
# then train — detects schema, trains both models, auto-selects the best:
python main.py

# serve the API + dashboard
uvicorn api.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

### API

| Endpoint | Description |
|---|---|
| `GET /health` | liveness |
| `GET /model-info` | approved model, metrics, detected schema |
| `POST /forecast` | 7-day forecast + business insight for a series |

```bash
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"history": [ {"Store":1,"Date":"2015-06-01","Sales":5200, ...}, ... ]}'
```

---

## 🗺️ Roadmap

This is v1 of a forecasting platform. The architecture is built for the full vision:

**✅ Phase 1 — Production MVP (this release)**
Generic pipeline · auto schema detection · XGBoost + LSTM · auto-selection ·
MLflow-ready · FastAPI · dashboard · Docker · K8s manifests · HTTPS-ready.

**🔜 Phase 2 — Monitoring & Automation**
Evidently AI drift monitoring · model-health dashboard · Apache Airflow retraining DAG
(ingest → validate → drift-check → retrain-if-needed → evaluate → register → deploy) ·
automatic model registration.

**🔮 Phase 3 — Self-Serve Platform**
CSV upload → automatic dataset profiling → schema detection → preprocessing → feature
engineering → model selection → forecast, all hands-free. Multi-domain support
(retail, energy, logistics, restaurants, supply chain) and AutoML-style workflow.

> **Deployment note:** Airflow (Phase 2) is intended for local orchestration/demo; the
> production instance serves only the API. This mirrors real setups where orchestration
> runs separately from the serving layer.

---

## 📁 Structure

```
salescast/
├── src/forecasting/
│   ├── interfaces/contracts.py      # all ABCs + data contracts
│   ├── core/registry.py             # pluggable-model registry
│   ├── implementations/             # concrete, swappable components
│   │   ├── loaders/  profilers/  preprocessors/  models/  evaluators/
│   └── pipeline/                    # training + forecaster orchestration
├── api/main.py                      # FastAPI service
├── dashboard/index.html             # retail-analytics UI
├── config/config.yaml               # data path, horizon, models, overrides
├── tests/                           # interface smoke tests
├── main.py  ·  Dockerfile  ·  requirements.txt
└── K8s/                             # deployment + service (Phase 1)
```

---

## 👤 Author

**Omar Hatem** — ML / MLOps Engineer · Cairo, Egypt
[GitHub](https://github.com/omarhatem44) · [LinkedIn](https://www.linkedin.com/in/omar-h-mohamed-355ba4369/)

---

<div align="center">

*A reusable, model-agnostic demand-forecasting platform — architected for extension, delivered in phases.*

</div>