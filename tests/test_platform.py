"""Interface smoke tests.

These check the contracts the platform is built on rather than the accuracy of
any one model: that every implementation satisfies its ABC, that the registry
allows a new model without touching the pipeline, that schema detection works on
a dataset with no retail column names, and that drift detection separates a
stable distribution from a shifted one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forecasting.core.registry import available_models, get_model, register_model
from forecasting.implementations.evaluators.standard import (
    BestByMetricSelector,
    StandardEvaluator,
)
from forecasting.implementations.loaders.data_loaders import (
    BasicProfiler,
    CsvLoader,
    DataFrameLoader,
)
from forecasting.implementations.models import forecasters  # noqa: F401  (registers)
from forecasting.implementations.monitors.drift import (
    EvidentlyDriftMonitor,
    SimpleDriftMonitor,
)
from forecasting.implementations.preprocessors.generic import (
    CalendarFeatureEngineer,
    GenericPreprocessor,
    SlidingWindowBuilder,
)
from forecasting.implementations.profilers.schema_detector import HeuristicSchemaDetector
from forecasting.interfaces.contracts import (
    EvalResult,
    IDataLoader,
    IDataProfiler,
    IFeatureEngineer,
    IModel,
    IModelEvaluator,
    IModelSelector,
    IMonitor,
    IPreprocessor,
    ISchemaDetector,
    IWindowBuilder,
)


# -- fixtures --------------------------------------------------
@pytest.fixture
def retail_frame() -> pd.DataFrame:
    """Retail-flavoured frame: the target is findable by name hint."""
    rng = np.random.default_rng(0)
    days = pd.date_range("2013-01-01", periods=200, freq="D")
    frames = []
    for store, stype in ((1, "a"), (2, "b")):
        frames.append(pd.DataFrame({
            "Store": store,
            "Date": days.astype(str),
            "Sales": rng.normal(6000, 500, len(days)).round(),
            "Promo": rng.integers(0, 2, len(days)),
            "DayOfWeek": days.dayofweek + 1,
            "StoreType": stype,
        }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def energy_frame() -> pd.DataFrame:
    """No retail words anywhere: proves the detector is not domain-specific.

    The target has to be found by variance, not by a name hint.
    """
    rng = np.random.default_rng(1)
    n = 400
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="D").astype(str),
        "load_mw": rng.normal(900, 120, n),
        "temperature_c": rng.normal(20, 2, n),
        "is_holiday": rng.integers(0, 2, n),
    })


# -- contracts -------------------------------------------------
@pytest.mark.parametrize("impl, interface", [
    (CsvLoader("x.csv"), IDataLoader),
    (DataFrameLoader(pd.DataFrame()), IDataLoader),
    (BasicProfiler(), IDataProfiler),
    (HeuristicSchemaDetector(), ISchemaDetector),
    (GenericPreprocessor(), IPreprocessor),
    (CalendarFeatureEngineer(), IFeatureEngineer),
    (SlidingWindowBuilder(), IWindowBuilder),
    (StandardEvaluator(), IModelEvaluator),
    (BestByMetricSelector(), IModelSelector),
    (SimpleDriftMonitor(), IMonitor),
    (EvidentlyDriftMonitor(), IMonitor),
])
def test_implementations_satisfy_their_interface(impl, interface):
    assert isinstance(impl, interface)


def test_registered_models_implement_imodel():
    assert {"xgboost", "lstm"} <= set(available_models())
    for name in available_models():
        assert issubclass(get_model(name), IModel)


# -- registry (Open/Closed) ------------------------------------
def test_new_model_registers_without_touching_the_pipeline():
    @register_model("dummy_test_model")
    class _Dummy(IModel):
        def fit(self, X, y): return self
        def predict(self, X): return np.zeros((len(X), 1))
        def save(self, path): pass
        @classmethod
        def load(cls, path): return cls()

    assert "dummy_test_model" in available_models()
    assert get_model("dummy_test_model") is _Dummy
    assert _Dummy.name == "dummy_test_model"


def test_unknown_model_raises_with_available_names():
    with pytest.raises(KeyError) as exc:
        get_model("prophet")
    assert "Available" in str(exc.value)


# -- schema detection ------------------------------------------
def test_detects_schema_on_retail_data(retail_frame):
    schema = HeuristicSchemaDetector().detect(retail_frame)
    assert schema.date_column == "Date"
    assert schema.target_column == "Sales"
    assert schema.group_column == "Store"
    assert schema.frequency == "D"


def test_detects_schema_with_no_retail_column_names(energy_frame):
    """The platform claim is domain-independence, so this is the test that matters."""
    schema = HeuristicSchemaDetector().detect(energy_frame)
    assert schema.date_column == "timestamp"
    assert schema.target_column == "load_mw"       # highest variance numeric
    assert schema.frequency == "D"
    assert schema.confidence["target"] < 0.9       # guessed, and flagged as such
    assert any("verify" in n for n in schema.notes)


def test_group_column_is_none_for_a_single_series(energy_frame):
    """Regression: a low-cardinality categorical must not be mistaken for a
    series key when the dataset holds one series and dates are already unique."""
    schema = HeuristicSchemaDetector().detect(energy_frame)
    assert schema.group_column is None


def test_group_column_is_not_a_plain_categorical(retail_frame):
    """StoreType matches the ID name hint but does not make (group, date) unique."""
    schema = HeuristicSchemaDetector().detect(retail_frame)
    assert schema.group_column == "Store"


def test_config_overrides_beat_detection(energy_frame):
    schema = HeuristicSchemaDetector().detect(
        energy_frame, overrides={"target_column": "temperature_c"}
    )
    assert schema.target_column == "temperature_c"
    assert "temperature_c" not in schema.numerical_features


def test_detector_raises_when_no_numeric_target_exists():
    df = pd.DataFrame({"Date": ["2020-01-01", "2020-01-02"], "label": ["a", "b"]})
    with pytest.raises(ValueError):
        HeuristicSchemaDetector().detect(df)


# -- profiling -------------------------------------------------
def test_profiler_flags_constant_columns_and_duplicates():
    df = pd.DataFrame({"a": [1, 1, 1, 1], "b": [5, 5, 6, 6]})
    profile = BasicProfiler().profile(pd.concat([df, df.tail(1)]))
    assert profile.n_rows == 5
    assert "a" in profile.constant_columns
    assert profile.duplicate_rows > 0
    assert profile.warnings


# -- evaluation and selection ----------------------------------
def test_rmspe_ignores_zero_actuals():
    class _Perfect(IModel):
        name = "perfect"
        def fit(self, X, y): return self
        def predict(self, X): return np.array([0.0, 100.0, 200.0])
        def save(self, path): pass
        @classmethod
        def load(cls, path): return cls()

    result = StandardEvaluator().evaluate(_Perfect(), None, np.array([0.0, 100.0, 200.0]))
    assert result.rmspe == pytest.approx(0.0, abs=1e-6)
    assert result.mae == pytest.approx(0.0)


def test_selector_picks_the_lowest_metric():
    results = [
        EvalResult(model_name="lstm", mae=9, rmse=9, rmspe=0.30),
        EvalResult(model_name="xgboost", mae=5, rmse=5, rmspe=0.12),
    ]
    assert BestByMetricSelector("rmspe").select(results) == "xgboost"


def test_selector_falls_back_when_metric_is_all_nan():
    results = [
        EvalResult(model_name="a", mae=9, rmse=9.0, rmspe=float("nan")),
        EvalResult(model_name="b", mae=5, rmse=4.0, rmspe=float("nan")),
    ]
    assert BestByMetricSelector("rmspe").select(results) == "b"


# -- drift monitoring ------------------------------------------
def test_psi_reports_no_drift_on_a_stable_distribution():
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 3000)})
    cur = pd.DataFrame({"x": rng.normal(0, 1, 3000)})
    report = SimpleDriftMonitor().check_drift(ref, cur)
    assert report["dataset_drift"] is False
    assert report["columns"]["x"]["psi"] < 0.1


def test_psi_detects_a_shifted_distribution():
    rng = np.random.default_rng(3)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 3000)})
    cur = pd.DataFrame({"x": rng.normal(4, 1, 3000)})
    report = SimpleDriftMonitor().check_drift(ref, cur)
    assert report["dataset_drift"] is True
    assert report["columns"]["x"]["psi"] > 0.25


def test_evidently_monitor_always_returns_a_usable_report():
    """Whether Evidently is installed or not, the health endpoint must not break."""
    rng = np.random.default_rng(4)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 500)})
    cur = pd.DataFrame({"x": rng.normal(3, 1, 500)})
    report = EvidentlyDriftMonitor().check_drift(ref, cur)
    assert set(report) >= {"method", "drift_share", "dataset_drift"}
    assert isinstance(report["dataset_drift"], bool)


# -- model round-trip ------------------------------------------
def test_xgboost_forecaster_round_trips_through_disk(tmp_path):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(120, 6))
    y = rng.normal(size=(120, 3))                 # horizon = 3

    model = get_model("xgboost")(horizon=3).fit(X, y)
    before = model.predict(X)
    assert before.shape == (120, 3)

    path = tmp_path / "model.joblib"
    model.save(str(path))
    after = type(model).load(str(path)).predict(X)
    np.testing.assert_allclose(before, after)


def test_lstm_trains_on_large_magnitude_targets(tmp_path):
    """Regression: unscaled targets in the thousands left the LSTM predicting ~0,
    which made the XGBoost-vs-LSTM comparison meaningless. RMSPE near 1.0 means
    the network never left zero, so this pins it well below that.
    """
    pytest.importorskip("tensorflow")
    rng = np.random.default_rng(6)
    n, lookback, horizon = 300, 14, 7
    level = 6000 + rng.normal(0, 300, n + lookback + horizon).cumsum() * 0.1

    X = np.stack([level[i:i + lookback].reshape(-1, 1) for i in range(n)]).astype("float32")
    y = np.stack([level[i + lookback:i + lookback + horizon] for i in range(n)]).astype("float32")

    model = get_model("lstm")(horizon=horizon, epochs=30).fit(X, y)
    pred = model.predict(X)

    assert pred.shape == (n, horizon)
    # predictions must sit in the same order of magnitude as the target
    assert 0.3 * y.mean() < pred.mean() < 3 * y.mean()
    assert StandardEvaluator().evaluate(model, X, y).rmspe < 0.6

    path = tmp_path / "lstm.keras"
    model.save(str(path))
    reloaded = type(model).load(str(path))
    np.testing.assert_allclose(reloaded.predict(X), pred, rtol=1e-4)


# -- end-to-end wiring -----------------------------------------
def test_preprocess_engineer_window_produces_trainable_arrays(retail_frame):
    schema = HeuristicSchemaDetector().detect(retail_frame)
    df = GenericPreprocessor().fit_transform(retail_frame, schema)
    df = CalendarFeatureEngineer().engineer(df, schema)
    X_tab, X_seq, y, feat_cols = SlidingWindowBuilder(lookback=14, horizon=7).build(df, schema)

    assert len(X_tab) == len(X_seq) == len(y) > 0
    assert y.shape[1] == 7
    assert X_seq.shape[1] == 14
    assert not np.isnan(X_tab).any()