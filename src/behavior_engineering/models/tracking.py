"""Registro do treino do pipeline no MLflow, ligado às versões do Git e do DVC."""

import math
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import yaml
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from behavior_engineering.config import Params, Settings

RUN_NAME = "XGBoost_Pipeline"
UNKNOWN = "unknown"
CHAMPION_ALIAS = "champion"
PROMOTION_METRIC = "test_pr_auc"
# Sem o XGBoost nesta lista, o formato padrão (skops) recusa salvar o modelo
SKOPS_TRUSTED_TYPES = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]


def flatten_params(params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Achata dicionários aninhados em chaves com ponto (ex.: `model.max_depth`)."""
    flat: dict[str, Any] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_params(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def read_data_md5(processed_dir: Path, lock_path: Path = Path("dvc.lock")) -> str:
    """Lê do `dvc.lock` o hash de `processed_dir`, gravado ao fim do estágio `preprocess`."""
    if not lock_path.exists():
        return UNKNOWN
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    outs = lock.get("stages", {}).get("preprocess", {}).get("outs", [])
    for out in outs:
        if Path(out["path"]) == processed_dir:
            return out["md5"]
    return UNKNOWN


def read_git_dirty() -> str:
    """Indica se há alterações locais; o MLflow não registra isso, só o commit atual."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):  # fora de um repositório (ex.: Docker)
        return UNKNOWN
    return str(bool(result.stdout.strip())).lower()


class TrainingTracker:
    """Registra um treino no MLflow; com `mlflow_enabled=False`, não faz nada."""

    def __init__(self, settings: Settings) -> None:
        self.enabled = settings.mlflow_enabled
        self.tracking_uri = settings.mlflow_tracking_uri
        self.experiment_name = settings.mlflow_experiment_name
        self.registered_model_name = settings.mlflow_registered_model_name

    @contextmanager
    def run(self, processed_dir: Path) -> Iterator[None]:
        """Abre o run antes do treino, para um servidor fora do ar falhar logo no início."""
        if not self.enabled:
            yield
            return
        # O padrão do MLflow leva cerca de 4 minutos para desistir de um servidor fora do ar
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        tags = {"git_dirty": read_git_dirty(), "data_md5": read_data_md5(processed_dir)}
        with mlflow.start_run(run_name=RUN_NAME, tags=tags):
            yield

    def log(
        self,
        params: Params,
        metrics: dict[str, float],
        pipeline: Pipeline,
        input_example: pd.DataFrame,
    ) -> None:
        """Registra parâmetros, métricas e o modelo no run aberto, e promove se for o melhor."""
        if not self.enabled:
            return
        mlflow.log_params(flatten_params(params.model_dump(mode="json")))
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=input_example,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
            registered_model_name=self.registered_model_name,
        )
        self._promote_if_better(model_info.registered_model_version, metrics[PROMOTION_METRIC])

    def _champion_metric(self, client: MlflowClient) -> float:
        """Lê a métrica de promoção da versão campeã; sem campeã, qualquer versão a supera."""
        registered_model = client.get_registered_model(self.registered_model_name)
        version = registered_model.aliases.get(CHAMPION_ALIAS)
        if version is None:
            return -math.inf
        run_id = client.get_model_version(self.registered_model_name, version).run_id
        return client.get_run(run_id).data.metrics.get(PROMOTION_METRIC, -math.inf)

    def _promote_if_better(self, version: int, metric: float) -> None:
        """Aponta o alias da campeã para a versão, se ela superar a atual na métrica."""
        client = MlflowClient()
        champion_metric = self._champion_metric(client)
        if metric <= champion_metric:
            print(f"Versão {version} não promovida: {metric:.4f} <= {champion_metric:.4f}")
            return
        client.set_registered_model_alias(self.registered_model_name, CHAMPION_ALIAS, version)
        print(f"Versão {version} promovida a {CHAMPION_ALIAS}: {metric:.4f}")
