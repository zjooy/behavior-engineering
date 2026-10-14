import json
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from behavior_engineering.models.evaluator import Evaluator

METRIC_NAMES = {"test_accuracy", "test_f1", "test_precision", "test_recall", "test_pr_auc"}


def make_fitted_pipeline() -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    features = pd.DataFrame({"PageValues": [0.0, 1.0, 2.0, 3.0]})
    target = pd.Series([0, 1, 0, 1])
    pipeline = Pipeline([("classifier", DummyClassifier(strategy="stratified", random_state=42))])
    return pipeline.fit(features, target), features, target


def test_evaluate_returns_all_metrics_between_zero_and_one() -> None:
    pipeline, features, target = make_fitted_pipeline()
    metrics = Evaluator().evaluate(pipeline, features, target)

    assert set(metrics) == METRIC_NAMES
    assert all(0.0 <= value <= 1.0 for value in metrics.values())


def test_save_writes_valid_json(tmp_path: Path) -> None:
    metrics = {"test_pr_auc": 0.75}
    path = tmp_path / "nested" / "metrics.json"
    Evaluator().save(metrics, path)

    assert json.loads(path.read_text(encoding="utf-8")) == metrics
