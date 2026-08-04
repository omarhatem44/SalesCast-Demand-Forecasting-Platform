# SalesCast — Airflow Retraining (Phase 2)

Local Apache Airflow (2.10.5) running the **conditional retraining DAG** for the
SalesCast platform. Airflow runs locally for orchestration/demo; the production
EC2 instance only serves the API (this mirrors real setups where orchestration
is separate from serving).

## The DAG: `salescast_retraining`

```
data_ingestion → data_validation → feature_engineering → drift_detection → decide_retrain
                                                                                │
                                              ┌─────────────────────────────────┴──────────┐
                                         drift detected                                 no drift
                                              │                                            │
                                        retrain_model                                 skip_retrain
                                              │                                            │
                                         evaluation                                        │
                                              │                                            │
                                       register_in_mlflow                                  │
                                              │                                            │
                                              └──────────────────┬─────────────────────────┘
                                                            deploy_model
```

**Key feature — conditional retraining:** a `BranchPythonOperator` inspects the
drift-detection result and only runs `retrain_model` when drift is detected;
otherwise it skips straight to keeping the current model. This avoids needless
retraining and demonstrates conditional orchestration.

DAG tasks import the **same platform code** as the app (mounted at
`/opt/salescast`) — the pipeline and drift monitor are reused, not duplicated.

## Run it

Requires Docker Desktop (allocate ≥4GB RAM to Docker).

```bash
cd airflow
docker compose build          # builds Airflow image + platform deps
docker compose up airflow-init   # one-time DB + admin user
docker compose up -d          # start webserver + scheduler
```

Open **http://localhost:8080** — login `admin` / `admin`.
Enable the `salescast_retraining` DAG and trigger it.

Stop:
```bash
docker compose down            # add -v to also wipe the metadata DB
```

## Notes
- `register_in_mlflow` and `deploy_model` are stubs that log intent — wire them to
  a real MLflow registry / redeploy trigger for full production use.
- The DAG is scheduled `@weekly` but can be triggered manually from the UI.