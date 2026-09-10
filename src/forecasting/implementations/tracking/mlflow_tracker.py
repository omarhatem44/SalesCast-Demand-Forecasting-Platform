"""Experiment tracking and model registry.

Two implementations behind one interface, following the same defensive pattern
as the drift monitor: MLflowTracker is the real path, NullTracker is used when
MLflow is not installed or the tracking server is unreachable, so training never
fails because a side-channel is down.

Registry promotion is champion/challenger, not automatic: a newly trained model
is registered as a version every time, but it is only aliased to Production when
it beats the current Production model on the selection metric. That gate is the
reason to have a registry at all — without it, "registered" just means "saved
somewhere else as well".
"""
from __future__ import annotations

import os
from typing import Any

from forecasting.interfaces.contracts import EvalResult, ITracker



def _make_pyfunc_wrapper(model_name: str):
    """Wrap a platform model as an MLflow pyfunc so the registered version is
    genuinely loadable, not just a file parked next to a run. The wrapper
    reloads through the platform's own registry, so nothing about the model
    classes has to change to be servable from MLflow.
    """
    import mlflow

    class _SalesCastModel(mlflow.pyfunc.PythonModel):
        def __init__(self, name: str):
            self.name = name

        def load_context(self, context):
            from forecasting.core.registry import get_model
            from forecasting.implementations.models import forecasters  # noqa: F401
            self._model = get_model(self.name).load(context.artifacts["model_file"])

        def predict(self, context, model_input, params=None):
            import numpy as np
            return self._model.predict(np.asarray(model_input))

    return _SalesCastModel(model_name)


class NullTracker(ITracker):
    """No-op tracker. Keeps the pipeline working with no MLflow present."""

    def start_run(self, run_name: str, params: dict[str, Any]) -> None:
        pass

    def log_results(self, results: list[EvalResult]) -> None:
        pass

    def register_best(self, model_name: str, artifact_path: str,
                      metric_name: str, metric_value: float) -> dict | None:
        return None

    def end_run(self) -> None:
        pass

    @property
    def active(self) -> bool:
        return False


class MLflowTracker(ITracker):
    """Logs params, per-model metrics and artifacts; registers the winner.

    The default tracking URI is a local SQLite file because the MLflow file
    store does not implement the model registry — a file:// URI gives you runs
    but silently no versions, which is exactly the gap this class exists to
    close. Override with MLFLOW_TRACKING_URI to point at a real server.
    """

    def __init__(self, experiment: str = "salescast",
                 tracking_uri: str | None = None,
                 registered_model: str = "salescast-forecaster"):
        self.registered_model = registered_model
        self._run = None
        self._client = None
        self._ok = False

        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            uri = tracking_uri or os.environ.get(
                "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
            )
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
            self._mlflow = mlflow
            self._client = MlflowClient()
            self._ok = True
        except Exception as e:  # noqa: BLE001
            print(f"[mlflow] disabled ({type(e).__name__}: {str(e)[:120]})")

    @property
    def active(self) -> bool:
        return self._ok

    def start_run(self, run_name: str, params: dict[str, Any]) -> None:
        if not self._ok:
            return
        self._run = self._mlflow.start_run(run_name=run_name)
        self._mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_results(self, results: list[EvalResult]) -> None:
        """One metric per model per measure, so runs are comparable in the UI."""
        if not self._ok or self._run is None:
            return
        for r in results:
            self._mlflow.log_metrics({
                f"{r.model_name}_mae": r.mae,
                f"{r.model_name}_rmse": r.rmse,
                f"{r.model_name}_rmspe": r.rmspe,
            })

    def log_artifacts(self, path: str) -> None:
        if self._ok and self._run is not None and os.path.exists(path):
            self._mlflow.log_artifacts(path, artifact_path="artifacts")

    def register_best(self, model_name: str, artifact_path: str,
                      metric_name: str, metric_value: float) -> dict | None:
        """Register the trained model as a new version and gate promotion.

        Returns a dict describing the version and whether it was promoted, or
        None when tracking is unavailable.
        """
        if not self._ok or self._run is None:
            return None
        try:
            self._mlflow.log_metric("selected_metric_value", metric_value)
            self._mlflow.set_tag("selected_model", model_name)
            self._mlflow.set_tag("selection_metric", metric_name)

            info = self._mlflow.pyfunc.log_model(
                name="model",
                python_model=_make_pyfunc_wrapper(model_name),
                artifacts={"model_file": artifact_path},
                code_paths=[os.path.join("src", "forecasting")],
            )
            version = self._mlflow.register_model(info.model_uri, self.registered_model)
            self._client.set_model_version_tag(
                self.registered_model, version.version, metric_name, str(metric_value)
            )

            promoted, champion = self._maybe_promote(
                version.version, metric_name, metric_value
            )
            return {
                "version": int(version.version),
                "promoted": promoted,
                "champion_value": champion,
                "metric": metric_name,
                "value": metric_value,
            }
        except Exception as e:  # noqa: BLE001
            print(f"[mlflow] registration skipped ({type(e).__name__}: {str(e)[:120]})")
            return None

    def _maybe_promote(self, new_version: str, metric_name: str,
                       metric_value: float) -> tuple[bool, float | None]:
        """Promote only if the challenger beats the champion (lower is better)."""
        champion_value = None
        try:
            current = self._client.get_model_version_by_alias(
                self.registered_model, "Production"
            )
            tag = current.tags.get(metric_name)
            champion_value = float(tag) if tag is not None else None
        except Exception:  # noqa: BLE001
            current = None                       # no champion yet

        if current is None or champion_value is None or metric_value < champion_value:
            self._client.set_registered_model_alias(
                self.registered_model, "Production", new_version
            )
            return True, champion_value
        return False, champion_value

    def end_run(self) -> None:
        if self._ok and self._run is not None:
            self._mlflow.end_run()
            self._run = None


def build_tracker(cfg: dict) -> ITracker:
    """Config decides whether tracking is on; the pipeline stays unaware of which."""
    if not cfg.get("mlflow", {}).get("enabled", False):
        return NullTracker()
    mc = cfg["mlflow"]
    return MLflowTracker(
        experiment=mc.get("experiment", "salescast"),
        tracking_uri=mc.get("tracking_uri"),
        registered_model=mc.get("registered_model", "salescast-forecaster"),
    )