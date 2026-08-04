FROM apache/airflow:2.10.5
# Install the platform's runtime deps so DAG tasks can train/monitor.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow
RUN pip install --no-cache-dir \
    pandas numpy scikit-learn xgboost joblib pyyaml