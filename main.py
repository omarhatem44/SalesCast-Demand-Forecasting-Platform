"""Train the SalesCast platform end-to-end from config.

    python main.py                 # uses config/config.yaml
"""
import sys
import yaml

sys.path.insert(0, "src")
from forecasting.pipeline.training import TrainingPipeline  # noqa: E402


def main():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    TrainingPipeline(cfg).run()


if __name__ == "__main__":
    main()