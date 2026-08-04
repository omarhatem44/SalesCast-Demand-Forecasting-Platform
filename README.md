<div align="center">

# 📈 SalesCast — Demand Forecasting Platform

**A reusable, model-agnostic forecasting platform** — not a single-model demo.
It detects a dataset's structure automatically, engineers features, trains and
compares multiple models, auto-selects the best, serves 7-day forecasts with
business recommendations, **monitors data drift in production**, and
**retrains automatically when drift is detected**.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Baseline-EB5E28?style=for-the-badge)](https://xgboost.ai)
[![TensorFlow](https://img.shields.io/badge/LSTM-TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Evidently](https://img.shields.io/badge/Evidently-Drift-37C0D8?style=for-the-badge)](https://evidentlyai.com)
[![Airflow](https://img.shields.io/badge/Airflow-Retraining-017CEE?style=for-the-badge&logo=apacheairflow&logoColor=white)](https://airflow.apache.org)
[![MLflow](https://img.shields.io/badge/MLflow-Registry-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Online-3DDC84?style=for-the-badge)](https://salescast.duckdns.org)

**🌐 Live demo: [salescast.duckdns.org](https://salescast.duckdns.org)**

</div>

---

## 📑 Table of Contents
- [Why this is a platform, not a project](#-why-this-is-a-platform-not-a-project)
- [Screenshots](#-screenshots)
- [Architecture & design decisions](#-architecture--design-decisions)
- [The techniques — what, why, how](#-the-techniques--what-why-and-how)
- [Phase 1 — the forecasting MVP](#-phase-1--the-forecasting-mvp)
- [Phase 2 — monitoring & automated retraining](#-phase-2--monitoring--automated-retraining)
- [Quick start](#-quick-start)
- [Project structure](#-project-structure)
- [Roadmap](#-roadmap)

---

## 🎯 Why this is a platform, not a project

Most forecasting projects hardcode one dataset's column names and train one model.
SalesCast is built around **SOLID interfaces** so every stage is replaceable, and the
pipeline works on **any** business time-series with minimal configuration:

<img src="src/assets/d75c93e3-0f33-47ca-b250-4e29c08e31c1.png" width="900" alt="SalesCast Archi"/>

**Rossmann retail sales is only the first dataset**. Nothing in the pipeline is
Rossmann-specific — the schema detector finds the date/target/feature roles
automatically, so the same code forecasts energy demand, restaurant orders, or
supply-chain volume.

---

## 📸 Screenshots

> Replace the placeholders below with your captured images (save them under `assets/`).

**Forecasting dashboard (live):**  store selector, 7-day forecast with confidence band, KPIs, inventory recommendations.

<img src="src/assets/Screenshot 2026-08-04 213050.png" width="900" alt="SalesCast dashboard"/>

**Model Health panel (Phase 2 drift monitoring):** last retrain, data drift, prediction drift, model version.

<img src="src/assets/Screenshot 2026-08-04 213755.png" width="400" alt="Model health panel"/>


**Airflow retraining DAG (conditional branching):** the graph view showing retraining skipped when no drift is detected.

<img src="airflow/Path.png" width="900" alt="Airflow DAG graph"/>

---

## 🏗️ Architecture & design decisions

Every major stage is an abstract interface (`abc.ABC`) with interchangeable
implementations, wired via a config-driven registry. **New models or preprocessors
can be added without modifying existing code** (Open/Closed Principle).

| Interface | Responsibility | Phase 1 implementation |
|---|---|---|
| `IDataLoader` | load raw data | `CsvLoader`, `DataFrameLoader` |
| `IDataProfiler` | statistics + quality flags | `BasicProfiler` |
| `ISchemaDetector` | auto-detect date/target/features | `HeuristicSchemaDetector` |
| `IPreprocessor` | clean / encode / scale (schema-driven) | `GenericPreprocessor` |
| `IFeatureEngineer` | calendar / lag / rolling features | `CalendarFeatureEngineer` |
| `IWindowBuilder` | sliding windows for sequences | `SlidingWindowBuilder` |
| `IModel` | fit / predict / save / load | `XGBoostForecaster`, `LSTMForecaster` |
| `IModelEvaluator` | MAE / RMSE / RMSPE | `StandardEvaluator` |
| `IModelSelector` | pick the best model | `BestByMetricSelector` |
| `IForecaster` | serve N-day forecast + insight | `Forecaster` |
| `IMonitor` | drift detection | `EvidentlyDriftMonitor` (+ PSI fallback) |
| `IRetrainer` | retrain decision | Airflow branching DAG |

**Why interfaces?** So the platform can grow. Adding Prophet later is:
```python
@register_model("prophet")
class ProphetForecaster(IModel): ...
# instantly available to the pipeline, selector, and API — zero other changes
```

---

## 🔬 The techniques — what, why, and how

This section explains each major technique, *why* it was chosen, and *how* it works here.

### 1. Automatic schema detection
- **What:** infers the date column, target, numeric/categorical features, group column, and frequency from any CSV.
- **Why:** a platform can't assume column names. Hardcoding `"Sales"` and `"Date"` would make it a Rossmann script, not a reusable system.
- **How:** heuristics — the date column is the one that parses to datetime for the most rows; the target matches name hints (`sales/revenue/demand/orders`) or falls back to the highest-variance numeric column; frequency is inferred from the median gap between sorted dates. A **config override** lets the user correct any uncertain guess.

### 2. XGBoost baseline + LSTM, with honest auto-selection
- **What:** two model families trained and compared; the better one is chosen automatically.
- **Why:** deep learning is not always better. On tabular time-series with strong calendar/lag signals, gradient boosting frequently **beats** LSTMs. Forcing an LSTM to "look advanced" would be dishonest engineering. Training both and reporting the winner demonstrates judgment.
- **How:** both implement the same `IModel` interface (multi-output for the 7-day horizon). XGBoost uses one regressor per horizon step; the LSTM is a sequence-to-vector network. `BestByMetricSelector` picks the lowest **RMSPE**. On this dataset XGBoost wins — and the system says so.

### 3. RMSPE as the selection metric (not accuracy)
- **What:** Root Mean Square Percentage Error.
- **Why:** forecasting is regression, so "accuracy" is meaningless. RMSPE is scale-independent (a 100-unit error matters more on a 200-unit day than a 5,000-unit day) and is the exact metric the Rossmann Kaggle competition used, making results comparable. MAE and RMSE are also reported for completeness.
- **How:** `RMSPE = sqrt(mean(((y_true - y_pred) / y_true)²))`, ignoring days where true sales are 0 (standard practice, since percentage error is undefined there).

### 4. Sliding-window supervised framing
- **What:** turn a raw time-series into `(X, y)` training pairs.
- **Why:** both tree and sequence models need supervised examples. A window of the last *N* days predicts the next *H* days.
- **How:** `SlidingWindowBuilder(lookback=14, horizon=7)` produces a flattened window for XGBoost and a 3D `(samples, timesteps, features)` tensor for the LSTM from the same data.

### 5. Calendar + lag + rolling features
- **What:** day-of-week, month, week-of-year, weekend flags, plus lagged and rolling-mean target values.
- **Why:** retail demand is highly seasonal (weekends, holidays) and autocorrelated (last week predicts this week). These features give even simple models strong signal.
- **How:** `CalendarFeatureEngineer` derives them generically from whatever the schema's date/target columns are — grouped per series when a group column (e.g. Store) exists.

### 6. FastAPI for serving
- **What:** the REST API that serves forecasts and health.
- **Why:** FastAPI is async, fast, and auto-generates OpenAPI docs — a modern standard for ML serving.
- **How:** `/forecast` accepts a series' recent history and returns the 7-day forecast + business insight; the API **always loads the currently approved model** (never a hardcoded one), so swapping the winner needs no API change.

### 7. Drift monitoring (Evidently AI + PSI fallback) — Phase 2
- **What:** detect when live data drifts from the training distribution.
- **Why:** models silently decay when the world changes (new promotions, seasonality shifts). Detecting drift is what turns a one-off model into a maintainable production system.
- **How:** the primary path uses **Evidently AI**; if Evidently isn't installed or its API version differs, the platform falls back to a built-in **PSI (Population Stability Index)** check — so monitoring never breaks. PSI compares binned distributions: `< 0.1` stable, `0.1–0.25` moderate, `> 0.25` significant drift. Training saves a `reference_sample.csv` as the baseline to compare against.

### 8. Conditional retraining with Airflow — Phase 2
- **What:** an orchestrated pipeline that retrains **only when drift is detected**.
- **Why:** retraining on every schedule wastes compute and can even hurt a healthy model. Retraining *because* of a signal (drift) is the mature pattern. Airflow provides scheduling, retries, observability, and a UI.
- **How:** a `BranchPythonOperator` reads the drift-detection result and routes to `retrain_model` (if drift) or `skip_retrain` (if not); both converge on `deploy_model`. See [Phase 2](#-phase-2--monitoring--automated-retraining).

### 9. Slim, separated serving image
- **What:** the deployed container excludes TensorFlow.
- **Why:** the served model is XGBoost, so TensorFlow (large, RAM-hungry) is dead weight in production. The LSTM code imports TF lazily, so it's never loaded at serve time.
- **How:** `Dockerfile.serve` installs `requirements-serve.txt` (no TF) — dramatically smaller image, essential on a free-tier VM.

---

## 🚀 Phase 1 — the forecasting MVP

Generic pipeline · auto schema detection · XGBoost + LSTM · auto-selection ·
FastAPI · retail-analytics dashboard · Docker · Kubernetes manifests · HTTPS.

**Modeling — honest, not dogmatic.** The platform trains both models, evaluates on
MAE/RMSE/RMSPE, and auto-selects the winner. On Rossmann, XGBoost beats the LSTM and
is chosen automatically — the system reports this truthfully rather than forcing DL.

**Dashboard.** A retail-analytics interface (not just a chart): store selector,
historical sales, 7-day forecast with confidence band, KPIs, and inventory
recommendations derived from the forecast trend.

> Inventory Recommendation (from forecast trend):
> Milk → increase 14% · Coffee → increase 10% · Bread → increase 4%

### API

| Endpoint | Description |
|---|---|
| `GET /health` | liveness |
| `GET /model-info` | approved model, metrics, detected schema |
| `POST /forecast` | 7-day forecast + business insight for a series |
| `GET /model-health` | *(Phase 2)* drift status + model metadata |
| `POST /model-health` | *(Phase 2)* drift check against posted recent data |

---

## 🔁 Phase 2 — monitoring & automated retraining

### Drift monitoring
The **Model Health** panel on the dashboard shows, in real time:
- **Last retrain** timestamp
- **Data drift** status + share
- **Prediction drift** status
- **Model version** and approved model

Backed by `EvidentlyDriftMonitor` with a PSI fallback (see technique #7). The
`/model-health` endpoint compares recent data against the saved training reference
and returns a status: *HEALTHY*, *DRIFT DETECTED — retraining recommended*, or
*MONITORING*.

### Airflow retraining DAG

`salescast_retraining` — a production-quality DAG with **conditional retraining**:

```
data_ingestion → data_validation → feature_engineering → drift_detection → decide_retrain
                                                                                │
                                          ┌─────────────────────────────────────┴───────────┐
                                     drift detected                                      no drift
                                          │                                                  │
                                    retrain_model                                       skip_retrain
                                          │                                                  │
                                     evaluation                                              │
                                          │                                                  │
                                  register_in_mlflow                                         │
                                          │                                                  │
                                          └──────────────────────┬───────────────────────────┘
                                                            deploy_model
```

**Key feature:** a `BranchPythonOperator` retrains **only when drift is detected**,
otherwise it skips straight to keeping the current model. This avoids needless
retraining and demonstrates conditional orchestration — not a linear script.

**Deployment note:** Airflow runs **locally** (Docker Compose) for orchestration and
demos; the production instance serves only the API. This mirrors real setups where
orchestration is separate from the serving layer.

**Run it:**
```bash
cd airflow
docker compose build
docker compose up airflow-init      # one-time DB + admin user
docker compose up -d
# open http://localhost:8080  (admin / admin)
```

---

## ⚡ Quick start

```bash
pip install -r requirements.txt

# drop the real Rossmann train.csv in data/ (or use the included synthetic sample)
python main.py                      # detects schema, trains both models, auto-selects

uvicorn api.main:app --host 0.0.0.0 --port 8000
# open http://localhost:8000
```

**Deploy (slim, TF-free serving image):**
```bash
docker build -f Dockerfile.serve -t salescast .
docker run -d --name salescast -p 8000:8000 salescast
```

---

## 📁 Project structure

```
salescast/
├── src/forecasting/
│   ├── interfaces/contracts.py          # all ABCs + data contracts
│   ├── core/registry.py                 # pluggable-model registry
│   ├── implementations/
│   │   ├── loaders/        profilers/   preprocessors/
│   │   ├── models/         evaluators/  monitors/    # monitors = Phase 2 drift
│   └── pipeline/
│       ├── training.py                  # end-to-end training orchestrator
│       ├── forecaster.py                # loads approved model, 7-day forecast
│       └── health.py                    # Phase 2 model-health service
├── api/main.py                          # FastAPI (forecast + health endpoints)
├── dashboard/index.html                 # retail-analytics UI + health panel
├── airflow/
│   ├── dags/salescast_retraining_dag.py # conditional retraining DAG
│   ├── docker-compose.yaml              # local Airflow (LocalExecutor + Postgres)
│   └── Dockerfile                       # Airflow image + platform deps
├── config/config.yaml                   # data path, horizon, models, overrides
├── K8s/                                 # deployment + service manifests
├── tests/test_platform.py               # interface smoke tests
├── main.py · Dockerfile · Dockerfile.serve
└── requirements.txt · requirements-serve.txt · requirements-monitoring.txt
```

---

## 🗺️ Roadmap

**✅ Phase 1 — Production MVP**
Generic pipeline · auto schema detection · XGBoost + LSTM · auto-selection ·
FastAPI · dashboard · Docker · K8s · HTTPS deployment.

**✅ Phase 2 — Monitoring & Automation**
Evidently/PSI drift monitoring · live model-health panel · Airflow retraining DAG
with conditional (drift-triggered) retraining.

**🔮 Phase 3 — Self-Serve Platform**
CSV upload → automatic profiling → schema detection → preprocessing → feature
engineering → model selection → forecast, hands-free. Multi-domain support
(retail, energy, logistics, restaurants, supply chain) and an AutoML-style workflow.
Full MLflow registry integration with champion/challenger promotion.

---

## 👤 Author

**Omar Hatem** — ML / MLOps Engineer · Cairo, Egypt
[GitHub](https://github.com/omarhatem44) · [LinkedIn](https://www.linkedin.com/in/omar-h-mohamed-355ba4369/)

---

<div align="center">

*A reusable, model-agnostic demand-forecasting platform with drift monitoring and
automated retraining — architected for extension, delivered in phases.*

</div>
