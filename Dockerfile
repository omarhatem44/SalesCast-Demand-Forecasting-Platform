# Training image — the full stack, including TensorFlow for the LSTM.
#
# Serving is a separate, slimmer image (Dockerfile.serve): the LSTM is only
# needed to train and compare, and the selected model on this data is XGBoost,
# so the runtime container has no reason to carry TensorFlow.
#
#   docker build -t salescast-train .
#   docker run --rm -v "$PWD/artifacts:/app/artifacts" salescast-train
#
FROM python:3.10-slim-bookworm

WORKDIR /app

# libgomp1 is required by XGBoost's OpenMP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

# Trains every model listed in config/config.yaml, evaluates them on the same
# split, selects the winner, and writes artifacts/ (model + summary.json).
CMD ["python", "main.py"]