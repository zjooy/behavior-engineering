"""Estágio `train`: treina o pipeline, avalia no teste e salva modelo e métricas."""

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from behavior_engineering.config import Params, Settings, load_params
from behavior_engineering.data.loader import DataLoader
from behavior_engineering.features.preprocessor import build_preprocessor
from behavior_engineering.models.evaluator import Evaluator
from behavior_engineering.models.tracking import TrainingTracker
from behavior_engineering.models.trainer import Trainer
from behavior_engineering.reproducibility import set_global_seed


def load_split(loader: DataLoader, path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Lê um split gerado pelo `preprocess` e o separa em (X, y)."""
    return loader.split_features_target(loader.load(path))


def fit_pipeline(params: Params, features: pd.DataFrame, target: pd.Series) -> Pipeline:
    """Monta e ajusta o pipeline (pré-processador + XGBoost) e o salva em disco."""
    trainer = Trainer(build_preprocessor(params.features), params.model, params.seed)
    pipeline = trainer.fit(features, target)
    trainer.save(params.train.model_path)
    return pipeline


def evaluate_pipeline(
    params: Params, pipeline: Pipeline, features: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    """Calcula as métricas no conjunto de teste e as grava em JSON."""
    evaluator = Evaluator()
    metrics = evaluator.evaluate(pipeline, features, target)
    evaluator.save(metrics, params.train.metrics_path)
    return metrics


def main() -> None:
    """Executa treino e avaliação a partir dos splits gerados pelo `preprocess`."""
    settings = Settings()
    params = load_params(settings.params_path)
    set_global_seed(params.seed)

    loader = DataLoader(params.data.target)
    x_train, y_train = load_split(loader, params.data.processed_dir / "train.csv")
    x_test, y_test = load_split(loader, params.data.processed_dir / "test.csv")

    tracker = TrainingTracker(settings)
    with tracker.run(params.data.processed_dir):
        pipeline = fit_pipeline(params, x_train, y_train)
        metrics = evaluate_pipeline(params, pipeline, x_test, y_test)
        tracker.log(params, metrics, pipeline, x_train.head())
    print(metrics)


if __name__ == "__main__":
    main()
