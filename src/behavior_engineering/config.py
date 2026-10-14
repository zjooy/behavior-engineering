"""Configuração do projeto: variáveis de ambiente (.env) e parâmetros (params.yaml)."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações dependentes do ambiente, lidas do `.env` ou de variáveis de ambiente."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    params_path: Path = Path("configs/params.yaml")
    device: str = "cpu"
    mlflow_tracking_uri: str = "http://127.0.0.1:5000"
    mlflow_experiment_name: str = "Online_Shoppers_Base"


class DataParams(BaseModel):
    """Localização dos dados e regras de divisão treino/teste."""

    raw_path: Path
    processed_dir: Path
    target: str
    test_size: float


class FeatureParams(BaseModel):
    """Colunas descartadas e categóricas usadas pelo pré-processador."""

    drop: list[str]
    categorical: list[str]
    ohe_min_frequency: float


class TrainParams(BaseModel):
    """Destino dos artefatos gerados pelo treino."""

    model_path: Path
    metrics_path: Path


class Params(BaseModel):
    """Parâmetros versionados do pipeline (espelham `configs/params.yaml`)."""

    seed: int
    data: DataParams
    features: FeatureParams
    model: dict[str, Any]
    train: TrainParams


def load_params(path: Path) -> Params:
    """Lê e valida o arquivo YAML de parâmetros."""
    with path.open(encoding="utf-8") as file:
        return Params.model_validate(yaml.safe_load(file))
