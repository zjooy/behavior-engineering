from pathlib import Path

import mlflow
import pandas as pd
import pytest
import yaml
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from behavior_engineering.config import Settings, load_params
from behavior_engineering.models.tracking import (
    CHAMPION_ALIAS,
    UNKNOWN,
    TrainingTracker,
    flatten_params,
    read_data_md5,
)

PARAMS_PATH = Path(__file__).parents[1] / "configs" / "params.yaml"
PROCESSED_DIR = Path("data/processed")


def make_fitted_pipeline() -> tuple[Pipeline, pd.DataFrame]:
    features = pd.DataFrame({"PageValues": [0.0, 1.0, 2.0, 3.0]})
    target = pd.Series([0, 1, 0, 1])
    pipeline = Pipeline([("classifier", XGBClassifier(n_estimators=2))])
    return pipeline.fit(features, target), features


def write_lock(path: Path) -> None:
    outs = [{"path": "data/processed", "md5": "abc.dir"}]
    path.write_text(yaml.safe_dump({"stages": {"preprocess": {"outs": outs}}}), encoding="utf-8")


def test_flatten_params_uses_dotted_keys() -> None:
    assert flatten_params({"seed": 42, "model": {"max_depth": 4}}) == {
        "seed": 42,
        "model.max_depth": 4,
    }


def test_read_data_md5_finds_processed_dir(tmp_path: Path) -> None:
    lock_path = tmp_path / "dvc.lock"
    write_lock(lock_path)

    assert read_data_md5(PROCESSED_DIR, lock_path) == "abc.dir"


def test_read_data_md5_without_lock_is_unknown(tmp_path: Path) -> None:
    assert read_data_md5(PROCESSED_DIR, tmp_path / "dvc.lock") == UNKNOWN


def test_tracker_logs_params_metrics_tags_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    write_lock(tmp_path / "dvc.lock")
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    settings = Settings(mlflow_tracking_uri=tracking_uri, mlflow_experiment_name="test")
    pipeline, features = make_fitted_pipeline()
    tracker = TrainingTracker(settings)

    with tracker.run(PROCESSED_DIR):
        tracker.log(load_params(PARAMS_PATH), {"test_pr_auc": 0.75}, pipeline, features)
        run_id = mlflow.active_run().info.run_id

    run = mlflow.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.metrics == {"test_pr_auc": 0.75}
    assert run.data.params["model.device"] == "cpu"
    assert run.data.tags["data_md5"] == "abc.dir"
    assert "git_dirty" in run.data.tags
    logged_models = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id], output_format="list"
    )
    assert len(mlflow.sklearn.load_model(logged_models[0].model_uri).predict(features)) == 4


def test_tracker_promotes_only_versions_that_beat_the_champion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    settings = Settings(mlflow_tracking_uri=tracking_uri, mlflow_experiment_name="test")
    pipeline, features = make_fitted_pipeline()
    tracker = TrainingTracker(settings)
    champions = []

    for pr_auc in [0.75, 0.75, 0.70, 0.80]:
        with tracker.run(PROCESSED_DIR):
            tracker.log(load_params(PARAMS_PATH), {"test_pr_auc": pr_auc}, pipeline, features)
        registered_model = MlflowClient().get_registered_model(tracker.registered_model_name)
        champions.append(registered_model.aliases[CHAMPION_ALIAS])

    assert champions == [1, 1, 1, 4]


def test_disabled_tracker_does_not_open_run() -> None:
    settings = Settings(mlflow_enabled=False, mlflow_tracking_uri="http://127.0.0.1:1")
    tracker = TrainingTracker(settings)
    pipeline, features = make_fitted_pipeline()

    with tracker.run(PROCESSED_DIR):
        tracker.log(load_params(PARAMS_PATH), {"test_pr_auc": 0.75}, pipeline, features)
        assert mlflow.active_run() is None
